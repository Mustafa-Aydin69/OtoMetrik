"""Faz 20-23: brand_model_median_price - Faz 19 (frequency_ablation.py) ablation
testinde (E grubu) dogrulanmis hiyerarsik fiyat ozelligi. B/C/D (ham
model_frequency/brand_model_frequency) REDDEDILDI - gain paylari %0.0-0.2,
tutarsiz/negatif nadir-model etkisi. E grubunun basarisinin neredeyse
TAMAMI brand_model_median_price'tan geliyordu (gain payi %31.7, digerleri
~%0.1 toplam) - bu yuzden SADECE bu ozellik production'a alinir.

FALLBACK ZINCIRI - Faz 23'te GUNCELLENDI (fallback_shrinkage_ablation.py'nin
B_model_tier sonucu ile, kullanicinin onayiyla), Faz 29'da DEGER HESAPLAMA
YONTEMI degisti (asagida) ama KATMAN SIRASI AYNI kaldi:
    marka+model medyani/egrisi (fold'da GORULMUSSE, count esigi YOK - tek
    ornekli bir grup bile kendi degerini alir)
    -> MODEL degeri (marka+model gorulmemis ama bu MODEL, HANGI MARKAYA ait
       olursa olsun, fold'da baska bir satirda gorulmusse - marka'dan
       BAGIMSIZ, tum markalar BIRLESTIRILEREK hesaplanir; bkz. _tier_curves())
    -> marka degeri (marka+model VE model hic gorulmemisse)
    -> global (train) degeri (marka da hic gorulmemisse)

Faz 29: DUZ MEDYAN yerine YAS-FARKINDALIKLI ROBUST EGRI (Theil-Sen). Kok neden:
preprocess.py'nin ESKI global q99 fiyat kirpmasi (Faz 9-28) Cadillac Escalade gibi
markalarda TESADUFEN sadece ESKI/yuksek-kilometreli satirlari hayatta birakiyordu
(yeni/dusuk-km olanlar global esigin ustundeydi) - duz medyan bu carpik yas
dagilimini yansitiyor, 2015 model bir Escalade icin bile "yasi ortalama 19 olan
bir grubun medyanini" donduruyordu (gercek fiyatin ~%70 altinda tahmin). preprocess.py
Faz 29'da marka-ici q99'a gecti (bkz. o modulun notu) ama bu FEATURE'in kendisi de
yas-kor kalmaya devam ediyordu - o yuzden burada da duzeltildi.

YONTEM: her katmanda (brand_model/model/brand/global) log(fiyat) ~ intercept +
slope*yas seklinde Theil-Sen (medyan-egim) robust regresyon fit edilir - OLS
DEGIL, cunku medyan kadar aykiri-degere dayanikli olmasi mevcut "medyan tabanli"
felsefeyi korur (bkz. _fit_curve()). Yeterli veri/yas cesitliligi YOKSA (n<3 veya
tum satirlar ayni yasta) DUZ (yas-kor) medyana geriler - slope=0, intercept=
log(medyan) - yani eski davranisla BIREBIR ayni deger uretir; hicbir katman daha
KOTU tahmin uretmez, sadece YETERLI veri oldugunda yas'i da hesaba katar.

SIZINTI ONLEME: egitim satirlari icin deger 5-fold OUT-OF-FOLD hesaplanir
(compute_oof_feature) - her satirin degeri, kendi fiyati DAHIL OLMADAN,
DIGER fold'lardan hesaplanir (bkz. frequency_ablation.py'deki ayni mantik,
Faz 19'da dogrulandi; MODEL katmani de AYNI fold_train alt kumesinden
hesaplanir - marka'dan bagimsiz olmasi sizinti onleme mantigini DEGISTIRMEZ,
sadece grupby anahtarini degistirir). Inference/serve icin (build_price_lookup)
TUM egitim verisinden (fold'suz) hesaplanir - gorulmemis (yeni) istekler zaten
bu hesaplamaya hic girmedigi icin sizinti riski YOK.

FIYAT REFERANS DONEMI UYARISI (kullanicinin metodolojik notu): bu ozellik
fiyat SEVIYESINE guclu bagimli (neredeyse target-encoding'e yakin, OOF
dogru uygulansa bile). Piyasa fiyatlari (enflasyon, arz/talep) zamanla
kayar - bu yuzden PRICE_REFERENCE_DATE artefakta ACIKCA kaydedilir. Faz 29:
bu tarih artik preprocess.PRICE_REFERENCE_DATE'ten (TEK zaman kaynagi -
"yas" ozelliginin egitimde hesaplandigi AYNI date.today() cagrisi) alinir -
onceden bu modul kendi ayri date.today() cagrisini yapiyordu, iki farkli
"simdi" kavrami ayni artefaktta yasayabiliyordu (bkz. preprocess.py Faz 29
notu). Model YENIDEN EGITILMEDEN uzun sure kullanilirsa bu lookup giderek
BAYATLAR; cozum yeniden egitimdir (bkz. reference_year/model_age_years'daki
ayni prensip, Faz 16). Faz 21'in zaman-bazli holdout'u (time_holdout_evaluation.py)
bu bayatlamayi henuz olcemedi (mevcut gercek kazima gecmisi sadece ~27 gun) -
haftalik/aylik otomatik retrain BASLATILMADI (kullanici karari, Faz 21 raporu).

VERI HASH'I (Faz 23, kullanicinin acik talebi): build_price_lookup() lookup'un
kuruldugu EGITIM verisinin (marka+model+fiyat) icerik hash'ini artefakta
kaydeder ('training_data_hash') - lookup'un HANGI veri anindan uretildigini
izlenebilir kilar (bkz. evaluate.py'deki dataset_hash ile ayni amac, farkli
kapsam: burada dosya degil, gercekten kullanilan DataFrame icerigi hash'lenir).
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocess import PRICE_REFERENCE_DATE

LOOKUP_VERSION = 3  # Faz 29: duz medyan -> yas-farkindalikli Theil-Sen egri
OOF_N_SPLITS = 5
OOF_SEED = 42  # frequency_ablation.py SEEDS[0] - ablation'da dogrulanan ilk/kanonik seed
FEATURE_COLUMN = 'brand_model_median_price'
FALLBACK_CHAIN = ['brand_model', 'model', 'brand', 'global']
MIN_CURVE_POINTS = 3  # bunun altinda Theil-Sen fit edilmez, duz medyana (slope=0) geriler
# theilslopes TUM ikili egimleri hesaplar (O(n^2) bellek/sure) - global/marka gibi
# genis katmanlarda (yuz binlerce satir) bu pratik olarak imkansiz. Genis katmanlarda
# sabit-seedli rastgele ALT ORNEKLEM uzerinden fit edilir - medyan-tabanli tahminciler
# (Theil-Sen dahil) alt orneklemede kararlidir, tam veriye ihtiyac duymaz; kucuk/ince
# gruplarda (asil duzeltmek istedigimiz nadir marka/model) zaten n bu esigin altinda
# kalir, TAM veriyle fit edilir.
MAX_CURVE_FIT_SAMPLE = 2000
CURVE_FIT_SAMPLE_SEED = 42


def _fit_curve(yas_values, fiyat_values):
    """(yas, fiyat) noktalarindan (intercept, slope, n) doner - log(fiyat) =
    intercept + slope*yas. Theil-Sen (medyan-egim) kullanir: OLS'den farkli
    olarak aykiri fiyatlara medyan kadar dayaniklidir (bkz. modul docstring'i).
    n<MIN_CURVE_POINTS veya tum yaslar ayniysa DUZ medyana (slope=0) geriler -
    boylece az veri/yas-cesitliligi olan gruplarda eski (Faz 20-28) davranisla
    BIREBIR ayni deger uretilir. NOT: doneri n, orijinal (alt orneklem oncesi)
    satir sayisidir - destek/guven raporlamasi icin gercek veri miktarini yansitir."""
    n = len(fiyat_values)
    distinct_ages = len(set(yas_values))
    if n >= MIN_CURVE_POINTS and distinct_ages >= 2:
        fit_yas, fit_fiyat = yas_values, fiyat_values
        if n > MAX_CURVE_FIT_SAMPLE:
            rng = np.random.default_rng(CURVE_FIT_SAMPLE_SEED)
            idx = rng.choice(n, size=MAX_CURVE_FIT_SAMPLE, replace=False)
            fit_yas, fit_fiyat = np.asarray(yas_values)[idx], np.asarray(fiyat_values)[idx]
        from scipy.stats import theilslopes  # egitim-zamani-only import (bkz. modul notu, serve.py bu fonksiyonu hic cagirmiyor)
        slope, intercept, _, _ = theilslopes(np.log(fit_fiyat), fit_yas)
        return float(intercept), float(slope), int(n)
    return float(np.log(np.median(fiyat_values))), 0.0, int(n)


def _eval_curve(intercept, slope, yas):
    return float(np.exp(intercept + slope * yas))


def _tier_curves(df):
    brand_model_curve = {
        key: _fit_curve(g['yas'].values, g['fiyat'].values)
        for key, g in df.groupby(['marka', 'model'], observed=True)
    }
    # MODEL katmani markadan BAGIMSIZ - ayni 'model' etiketine sahip TUM markalarin
    # satirlari BIRLESTIRILEREK fit edilir (bkz. modul docstring'i, Faz 23).
    model_curve = {
        key: _fit_curve(g['yas'].values, g['fiyat'].values)
        for key, g in df.groupby('model', observed=True)
    }
    brand_curve = {
        key: _fit_curve(g['yas'].values, g['fiyat'].values)
        for key, g in df.groupby('marka', observed=True)
    }
    global_curve = _fit_curve(df['yas'].values, df['fiyat'].values)
    return brand_model_curve, model_curve, brand_curve, global_curve


def _lookup(key_marka, key_model, query_yas, brand_model_curve, model_curve, brand_curve, global_curve):
    bm_key = (key_marka, key_model)
    if bm_key in brand_model_curve:
        intercept, slope, n = brand_model_curve[bm_key]
        return _eval_curve(intercept, slope, query_yas), 'brand_model', n
    if key_model in model_curve:
        intercept, slope, n = model_curve[key_model]
        return _eval_curve(intercept, slope, query_yas), 'model', n
    if key_marka in brand_curve:
        intercept, slope, n = brand_curve[key_marka]
        return _eval_curve(intercept, slope, query_yas), 'brand', n
    intercept, slope, n = global_curve
    return _eval_curve(intercept, slope, query_yas), 'global', n


def compute_oof_feature(X_full, y_full, n_splits=OOF_N_SPLITS, seed=OOF_SEED):
    """Egitim satirlari icin sizinti-siz (out-of-fold) brand_model_median_price.
    (values, sources) dondurur - sources sadece tanisal amacli (hangi fallback
    katmaninin kullanildigi)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X_full)
    values = np.empty(n, dtype=float)
    sources = np.empty(n, dtype=object)
    df = X_full[['marka', 'model', 'yas']].assign(fiyat=y_full.values)

    for fold_train_idx, fold_eval_idx in kf.split(df):
        fold_train = df.iloc[fold_train_idx]
        bm, mm, b, g = _tier_curves(fold_train)

        fold_eval = df.iloc[fold_eval_idx]
        rows = zip(fold_eval['marka'], fold_eval['model'], fold_eval['yas'])
        for pos, (marka, model, yas) in zip(fold_eval_idx, rows):
            val, src, _ = _lookup(marka, model, yas, bm, mm, b, g)
            values[pos] = val
            sources[pos] = src
    return values, sources


def _data_hash(df):
    """build_price_lookup()'un kuruldugu egitim verisinin (marka+model+fiyat) icerik
    hash'i - lookup'un HANGI veri anindan uretildigini izlenebilir kilar."""
    payload = pd.util.hash_pandas_object(df[['marka', 'model', 'fiyat']], index=False)
    return hashlib.md5(payload.values.tobytes()).hexdigest()[:12]


def _serialize_curve(curve):
    intercept, slope, n = curve
    return {'intercept': intercept, 'slope': slope, 'n': n}


def build_price_lookup(X_full, y_full):
    """Inference icin TUM (fold'suz) egitim verisinden hesaplanan lookup -
    model artefaktina 'hierarchical_price' anahtariyla gomulur."""
    df = X_full[['marka', 'model', 'yas']].assign(fiyat=y_full.values)
    brand_model_curve, model_curve, brand_curve, global_curve = _tier_curves(df)

    return {
        'lookup_version': LOOKUP_VERSION,
        'feature_column': FEATURE_COLUMN,
        'fallback_chain': list(FALLBACK_CHAIN),
        'oof_n_splits': OOF_N_SPLITS,
        'oof_seed': OOF_SEED,
        'min_curve_points': MIN_CURVE_POINTS,
        'brand_model_curve': {f'{k[0]}\x1f{k[1]}': _serialize_curve(v) for k, v in brand_model_curve.items()},
        'model_curve': {str(k): _serialize_curve(v) for k, v in model_curve.items()},
        'brand_curve': {str(k): _serialize_curve(v) for k, v in brand_curve.items()},
        'global_curve': _serialize_curve(global_curve),
        'price_reference_date': PRICE_REFERENCE_DATE.isoformat(),
        'training_data_hash': _data_hash(df),
        'normalization_notes': (
            'marka/model degerleri train.py CATEGORICAL_COLS ile ayni kanonik '
            'temsil (marka: category_mapping.py kanonik degerleri; model: '
            'serbest metin, normalize edilmemis) - ek bir normalizasyon '
            'uygulanmaz. "model" katmani markadan BAGIMSIZ (tum markalar '
            'birlestirilerek) hesaplanir (bkz. modul docstring, Faz 23). '
            'Deger = log(fiyat) ~ intercept + slope*yas Theil-Sen egrisi '
            '(Faz 29, bkz. modul docstring) - yeterli veri yoksa duz medyana '
            '(slope=0) geriler.'
        ),
    }


def lookup_price(marka, model, yas, lookup):
    """(deger, kaynak, destek_n) dondurur - kaynak in {'brand_model','model','brand','global'},
    destek_n o katmanin kac egitim satirindan hesaplandigi. Request basina DataFrame
    taramasi yok - sadece dict aramalari (bkz. hp_support.py ile ayni performans
    yaklasimi). Katman secimi DETERMINISTIK: brand_model bulunduysa -> brand_model;
    degilse model bulunduysa -> model; degilse marka bulunduysa -> brand; aksi
    halde -> global (bkz. FALLBACK_CHAIN)."""
    bm_dict = lookup['brand_model_curve']
    key = f'{marka}\x1f{model}'
    if key in bm_dict:
        c = bm_dict[key]
        return _eval_curve(c['intercept'], c['slope'], yas), 'brand_model', c['n']
    model_dict = lookup.get('model_curve', {})
    if str(model) in model_dict:
        c = model_dict[str(model)]
        return _eval_curve(c['intercept'], c['slope'], yas), 'model', c['n']
    if marka in lookup['brand_curve']:
        c = lookup['brand_curve'][marka]
        return _eval_curve(c['intercept'], c['slope'], yas), 'brand', c['n']
    c = lookup['global_curve']
    return _eval_curve(c['intercept'], c['slope'], yas), 'global', c['n']


def attach_oof_feature(X_full, y_full):
    values, sources = compute_oof_feature(X_full, y_full)
    return X_full.assign(**{FEATURE_COLUMN: values}), sources


def attach_lookup_feature(X, lookup):
    """Egitim-disi (holdout/inference) satirlar icin - fold'suz TAM egitim
    lookup'undan deger atar. X'in 'yas' kolonu ZATEN mevcut olmalidir (marka+model
    ile ayni asamada, load_clean_train_dataset()/load_cars1_holdout()/serve.py
    build_feature_row() tarafindan uretilir)."""
    keys_marka = X['marka'].astype(str)
    keys_model = X['model'].astype(str)
    keys_yas = X['yas']
    values = [
        lookup_price(m, mo, yas, lookup)[0]
        for m, mo, yas in zip(keys_marka, keys_model, keys_yas)
    ]
    return X.assign(**{FEATURE_COLUMN: values})
