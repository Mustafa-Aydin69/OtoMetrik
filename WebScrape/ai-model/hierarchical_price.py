"""Faz 20: brand_model_median_price - Faz 19 (frequency_ablation.py) ablation
testinde (E grubu) dogrulanmis TEK hiyerarsik fiyat ozelligi. B/C/D (ham
model_frequency/brand_model_frequency) REDDEDILDI - gain paylari %0.0-0.2,
tutarsiz/negatif nadir-model etkisi. E grubunun basarisinin neredeyse
TAMAMI brand_model_median_price'tan geliyordu (gain payi %31.7, digerleri
~%0.1 toplam) - bu yuzden SADECE bu ozellik production'a alinir.

FALLBACK ZINCIRI - ablation ile BIREBIR AYNI (KASITLI):
    marka+model medyani (fold'da GORULMUSSE, count esigi YOK - tek ornekli
    bir grup bile kendi medyanini alir, ablation'da da boyleydi)
    -> marka medyani (marka+model o fold'da hic gorulmemisse)
    -> global (train) medyani (marka da hic gorulmemisse)

NOT (kullanicinin acik talebi ile FARKLILIK): gorev tanimi "marka+model ->
MODEL -> marka -> global" seklinde 4 katmanli bir fallback oneriyor - ancak
frequency_ablation.py'deki E grubu SADECE 3 katmanli (marka+model -> marka
-> global) test edildi, "model" (marka'dan BAGIMSIZ, tum markalar birlesik)
katmani HICBIR ZAMAN olcumlenmedi. "Production implementasyonu deneyden
SAPMAMALI" talimatina uyarak buraya DOGRULANMAMIS bir 4. katman EKLENMEDI -
sadece ablation'da fiilen kullanilan 3 katman uygulanir. 4 katmanli surum
istenirse bu, kendi ablation deneyini gerektiren AYRI bir teklif olur.

SIZINTI ONLEME: egitim satirlari icin deger 5-fold OUT-OF-FOLD hesaplanir
(compute_oof_feature) - her satirin degeri, kendi fiyati DAHIL OLMADAN,
DIGER fold'lardan hesaplanir (bkz. frequency_ablation.py'deki ayni mantik,
Faz 19'da dogrulandi). Inference/serve icin (build_price_lookup) TUM egitim
verisinden (fold'suz) hesaplanir - gorulmemis (yeni) istekler zaten bu
hesaplamaya hic girmedigi icin sizinti riski YOK.

FIYAT REFERANS DONEMI UYARISI (kullanicinin metodolojik notu): bu ozellik
fiyat SEVIYESINE guclu bagimli (neredeyse target-encoding'e yakin, OOF
dogru uygulansa bile). Piyasa fiyatlari (enflasyon, arz/talep) zamanla
kayar - bu yuzden PRICE_REFERENCE_DATE artefakta ACIKCA kaydedilir. Model
YENIDEN EGITILMEDEN uzun sure kullanilirsa bu lookup giderek BAYATLAR;
cozum yeniden egitimdir (bkz. reference_year/model_age_years'daki ayni
prensip, Faz 16).
"""
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LOOKUP_VERSION = 1
OOF_N_SPLITS = 5
OOF_SEED = 42  # frequency_ablation.py SEEDS[0] - ablation'da dogrulanan ilk/kanonik seed
FEATURE_COLUMN = 'brand_model_median_price'


def _median_dicts(df):
    brand_model_median = df.groupby(['marka', 'model'], observed=True)['fiyat'].median().to_dict()
    brand_median = df.groupby('marka', observed=True)['fiyat'].median().to_dict()
    global_median = float(df['fiyat'].median())
    return brand_model_median, brand_median, global_median


def _lookup(key_marka, key_model, brand_model_median, brand_median, global_median):
    bm_key = (key_marka, key_model)
    if bm_key in brand_model_median:
        return brand_model_median[bm_key], 'brand_model'
    if key_marka in brand_median:
        return brand_median[key_marka], 'brand'
    return global_median, 'global'


def compute_oof_feature(X_full, y_full, n_splits=OOF_N_SPLITS, seed=OOF_SEED):
    """Egitim satirlari icin sizinti-siz (out-of-fold) brand_model_median_price.
    (values, sources) dondurur - sources sadece tanisal amacli (hangi fallback
    katmaninin kullanildigi)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X_full)
    values = np.empty(n, dtype=float)
    sources = np.empty(n, dtype=object)
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)

    for fold_train_idx, fold_eval_idx in kf.split(df):
        fold_train = df.iloc[fold_train_idx]
        bmm, bm, overall = _median_dicts(fold_train)

        fold_eval = df.iloc[fold_eval_idx]
        keys = list(zip(fold_eval['marka'], fold_eval['model']))
        for pos, key in zip(fold_eval_idx, keys):
            val, src = _lookup(key[0], key[1], bmm, bm, overall)
            values[pos] = val
            sources[pos] = src
    return values, sources


def build_price_lookup(X_full, y_full):
    """Inference icin TUM (fold'suz) egitim verisinden hesaplanan lookup -
    model artefaktina 'hierarchical_price' anahtariyla gomulur."""
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)
    brand_model_median, brand_median, global_median = _median_dicts(df)

    return {
        'lookup_version': LOOKUP_VERSION,
        'feature_column': FEATURE_COLUMN,
        'fallback_chain': ['brand_model', 'brand', 'global'],
        'oof_n_splits': OOF_N_SPLITS,
        'oof_seed': OOF_SEED,
        'brand_model_median': {f'{k[0]}\x1f{k[1]}': float(v) for k, v in brand_model_median.items()},
        'brand_median': {str(k): float(v) for k, v in brand_median.items()},
        'global_median': global_median,
        'price_reference_date': date.today().isoformat(),
        'normalization_notes': (
            'marka/model degerleri train.py CATEGORICAL_COLS ile ayni kanonik '
            'temsil (marka: category_mapping.py kanonik degerleri; model: '
            'serbest metin, normalize edilmemis) - ek bir normalizasyon '
            'uygulanmaz.'
        ),
    }


def lookup_price(marka, model, lookup):
    """(deger, kaynak) dondurur - kaynak in {'brand_model','brand','global'}.
    Request basina DataFrame taramasi yok - sadece dict aramalari (bkz.
    hp_support.py ile ayni performans yaklasimi)."""
    bm_dict = lookup['brand_model_median']
    key = f'{marka}\x1f{model}'
    if key in bm_dict:
        return bm_dict[key], 'brand_model'
    if marka in lookup['brand_median']:
        return lookup['brand_median'][marka], 'brand'
    return lookup['global_median'], 'global'


def attach_oof_feature(X_full, y_full):
    values, sources = compute_oof_feature(X_full, y_full)
    return X_full.assign(**{FEATURE_COLUMN: values}), sources


def attach_lookup_feature(X, lookup):
    """Egitim-disi (holdout/inference) satirlar icin - fold'suz TAM egitim
    lookup'undan deger atar."""
    keys_marka = X['marka'].astype(str)
    keys_model = X['model'].astype(str)
    values = [lookup_price(m, mo, lookup)[0] for m, mo in zip(keys_marka, keys_model)]
    return X.assign(**{FEATURE_COLUMN: values})
