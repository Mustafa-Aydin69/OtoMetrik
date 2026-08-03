"""Faz 20-23: brand_model_median_price - Faz 19 (frequency_ablation.py) ablation
testinde (E grubu) dogrulanmis hiyerarsik fiyat ozelligi. B/C/D (ham
model_frequency/brand_model_frequency) REDDEDILDI - gain paylari %0.0-0.2,
tutarsiz/negatif nadir-model etkisi. E grubunun basarisinin neredeyse
TAMAMI brand_model_median_price'tan geliyordu (gain payi %31.7, digerleri
~%0.1 toplam) - bu yuzden SADECE bu ozellik production'a alinir.

FALLBACK ZINCIRI - Faz 23'te GUNCELLENDI (fallback_shrinkage_ablation.py'nin
B_model_tier sonucu ile, kullanicinin onayiyla):
    marka+model medyani (fold'da GORULMUSSE, count esigi YOK - tek ornekli
    bir grup bile kendi medyanini alir)
    -> MODEL medyani (marka+model gorulmemis ama bu MODEL, HANGI MARKAYA ait
       olursa olsun, fold'da baska bir satirda gorulmusse - marka'dan
       BAGIMSIZ, tum markalar BIRLESTIRILEREK hesaplanir; bkz. _median_dicts())
    -> marka medyani (marka+model VE model hic gorulmemisse)
    -> global (train) medyani (marka da hic gorulmemisse)

Faz 20'de bu 4. katman ("model", markadan bagimsiz) hic olculmemisti (o zamanki
E grubu SADECE 3 katmanli test edilmisti) - bu yuzden Faz 20'de EKLENMEMISTI.
Faz 22'nin fallback_shrinkage_ablation.py'si bunu ayri bir ablation olarak
(B_model_tier) test etti: guclu-guclendirilmis segmentlerde (GENEL, nadir_model)
ciddi regresyon YOK, nadir modelde kucuk ama tekrar eden kazanc VAR (dis
holdout %-2.1, zaman-test %-1.9) - kullanici bunu production'a onayladi.
C_shrink varyanti (nadir_model'i dis holdout'ta %+13..+24 kotulestirdigi icin)
REDDEDILDI - fallback_shrinkage.py'de deneysel modul olarak kaliyor, production
pipeline'ina hic baglanmiyor.

SIZINTI ONLEME: egitim satirlari icin deger 5-fold OUT-OF-FOLD hesaplanir
(compute_oof_feature) - her satirin degeri, kendi fiyati DAHIL OLMADAN,
DIGER fold'lardan hesaplanir (bkz. frequency_ablation.py'deki ayni mantik,
Faz 19'da dogrulandi; MODEL katmani da AYNI fold_train alt kumesinden
hesaplanir - marka'dan bagimsiz olmasi sizinti onleme mantigini DEGISTIRMEZ,
sadece grupby anahtarini degistirir). Inference/serve icin (build_price_lookup)
TUM egitim verisinden (fold'suz) hesaplanir - gorulmemis (yeni) istekler zaten
bu hesaplamaya hic girmedigi icin sizinti riski YOK.

FIYAT REFERANS DONEMI UYARISI (kullanicinin metodolojik notu): bu ozellik
fiyat SEVIYESINE guclu bagimli (neredeyse target-encoding'e yakin, OOF
dogru uygulansa bile). Piyasa fiyatlari (enflasyon, arz/talep) zamanla
kayar - bu yuzden PRICE_REFERENCE_DATE artefakta ACIKCA kaydedilir. Model
YENIDEN EGITILMEDEN uzun sure kullanilirsa bu lookup giderek BAYATLAR;
cozum yeniden egitimdir (bkz. reference_year/model_age_years'daki ayni
prensip, Faz 16). Faz 21'in zaman-bazli holdout'u (time_holdout_evaluation.py)
bu bayatlamayi henuz olcemedi (mevcut gercek kazima gecmisi sadece ~27 gun) -
haftalik/aylik otomatik retrain BASLATILMADI (kullanici karari, Faz 21 raporu).

VERI HASH'I (Faz 23, kullanicinin acik talebi): build_price_lookup() lookup'un
kuruldugu EGITIM verisinin (marka+model+fiyat) icerik hash'ini artefakta
kaydeder ('training_data_hash') - lookup'un HANGI veri anindan uretildigini
izlenebilir kilar (bkz. evaluate.py'deki dataset_hash ile ayni amac, farkli
kapsam: burada dosya degil, gercekten kullanilan DataFrame icerigi hash'lenir).
"""
import hashlib
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LOOKUP_VERSION = 2  # Faz 23: 4 katmanli (marka+model->model->marka->global) zincire gecis
OOF_N_SPLITS = 5
OOF_SEED = 42  # frequency_ablation.py SEEDS[0] - ablation'da dogrulanan ilk/kanonik seed
FEATURE_COLUMN = 'brand_model_median_price'
FALLBACK_CHAIN = ['brand_model', 'model', 'brand', 'global']


def _median_dicts(df):
    brand_model_median = df.groupby(['marka', 'model'], observed=True)['fiyat'].median().to_dict()
    # MODEL medyani markadan BAGIMSIZ - ayni 'model' etiketine sahip TUM markalarin
    # satirlari BIRLESTIRILEREK hesaplanir (bkz. modul docstring'i, Faz 23).
    model_median = df.groupby('model', observed=True)['fiyat'].median().to_dict()
    brand_median = df.groupby('marka', observed=True)['fiyat'].median().to_dict()
    global_median = float(df['fiyat'].median())
    return brand_model_median, model_median, brand_median, global_median


def _lookup(key_marka, key_model, brand_model_median, model_median, brand_median, global_median):
    bm_key = (key_marka, key_model)
    if bm_key in brand_model_median:
        return brand_model_median[bm_key], 'brand_model'
    if key_model in model_median:
        return model_median[key_model], 'model'
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
        bmm, mm, bm, overall = _median_dicts(fold_train)

        fold_eval = df.iloc[fold_eval_idx]
        keys = list(zip(fold_eval['marka'], fold_eval['model']))
        for pos, key in zip(fold_eval_idx, keys):
            val, src = _lookup(key[0], key[1], bmm, mm, bm, overall)
            values[pos] = val
            sources[pos] = src
    return values, sources


def _data_hash(df):
    """build_price_lookup()'un kuruldugu egitim verisinin (marka+model+fiyat) icerik
    hash'i - lookup'un HANGI veri anindan uretildigini izlenebilir kilar."""
    payload = pd.util.hash_pandas_object(df[['marka', 'model', 'fiyat']], index=False)
    return hashlib.md5(payload.values.tobytes()).hexdigest()[:12]


def build_price_lookup(X_full, y_full):
    """Inference icin TUM (fold'suz) egitim verisinden hesaplanan lookup -
    model artefaktina 'hierarchical_price' anahtariyla gomulur."""
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)
    brand_model_median, model_median, brand_median, global_median = _median_dicts(df)

    return {
        'lookup_version': LOOKUP_VERSION,
        'feature_column': FEATURE_COLUMN,
        'fallback_chain': list(FALLBACK_CHAIN),
        'oof_n_splits': OOF_N_SPLITS,
        'oof_seed': OOF_SEED,
        'brand_model_median': {f'{k[0]}\x1f{k[1]}': float(v) for k, v in brand_model_median.items()},
        'model_median': {str(k): float(v) for k, v in model_median.items()},
        'brand_median': {str(k): float(v) for k, v in brand_median.items()},
        'global_median': global_median,
        'price_reference_date': date.today().isoformat(),
        'training_data_hash': _data_hash(df),
        'normalization_notes': (
            'marka/model degerleri train.py CATEGORICAL_COLS ile ayni kanonik '
            'temsil (marka: category_mapping.py kanonik degerleri; model: '
            'serbest metin, normalize edilmemis) - ek bir normalizasyon '
            'uygulanmaz. "model" katmani markadan BAGIMSIZ (tum markalar '
            'birlestirilerek) hesaplanir (bkz. modul docstring, Faz 23).'
        ),
    }


def lookup_price(marka, model, lookup):
    """(deger, kaynak) dondurur - kaynak in {'brand_model','model','brand','global'}.
    Request basina DataFrame taramasi yok - sadece dict aramalari (bkz.
    hp_support.py ile ayni performans yaklasimi). Katman secimi DETERMINISTIK:
    brand_model bulunduysa -> brand_model; degilse model bulunduysa -> model;
    degilse marka bulunduysa -> brand; aksi halde -> global (bkz. FALLBACK_CHAIN)."""
    bm_dict = lookup['brand_model_median']
    key = f'{marka}\x1f{model}'
    if key in bm_dict:
        return bm_dict[key], 'brand_model'
    model_dict = lookup.get('model_median', {})
    if str(model) in model_dict:
        return model_dict[str(model)], 'model'
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
