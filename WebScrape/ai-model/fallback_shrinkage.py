"""Faz 22: brand_model_median_price FALLBACK ZINCIRI icin ablation-only varyant
uretici. Faz 20/21'in ortaya cikardigi zayifligi (yeni marka-model kombinasyonlarinda
production'in baseline'a gore TUTARLI biçimde kotu tahmin etmesi, bkz. time_holdout_
evaluation.py) hedefler. Production hierarchical_price.py'ye HIC DOKUNMAZ - bu SADECE
fallback_shrinkage_ablation.py'nin kullandigi, ayri/deneysel bir modul.

3 varyant:
  A_current    : marka+model -> marka -> global (production'daki AYNI zincir)
  B_model_tier : marka+model -> model (markadan BAGIMSIZ) -> marka -> global
                 (kullanicinin Faz 20'deki ilk onerisi - o zaman hic olculmemisti)
  C_shrink_kX  : marka+model medyanini DESTEK SAYISINA gore marka medyanina dogru
                 "shrink" eder (Bayesian/James-Stein tarzi):
                     smoothed = (n * brand_model_medyan + k * marka_medyan) / (n + k)
                 n=marka+model grubunun (fold-train'deki) satir sayisi, k=shrinkage
                 sabiti - kucuk n'de marka medyanina yakin, buyuk n'de brand_model
                 medyanina yakin sonuc verir. marka da hic gorulmemisse (n_marka=0)
                 payda/pay dogrudan global'e duser (brand_model_median= global oldugu
                 icin formul otomatik global'e indirgenir).

SIZINTI ONLEME: hierarchical_price.py ile AYNI desen - egitim satirlari icin 5-fold
OUT-OF-FOLD (compute_oof_variant), inference/holdout icin fold'suz TAM train lookup'u
(build_variant_lookup + attach_variant_feature).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

OOF_N_SPLITS = 5
OOF_SEED = 42
VARIANTS = ['A_current', 'B_model_tier', 'C_shrink_k5', 'C_shrink_k20']


def _group_stats(df):
    """df: marka, model, fiyat kolonlari. Her katman icin (medyan, count) dondurur."""
    bm = df.groupby(['marka', 'model'], observed=True)['fiyat'].agg(['median', 'size'])
    model_stats = df.groupby('model', observed=True)['fiyat'].agg(['median', 'size'])
    brand_stats = df.groupby('marka', observed=True)['fiyat'].agg(['median', 'size'])
    global_median = float(df['fiyat'].median())
    return (bm['median'].to_dict(), bm['size'].to_dict(),
            model_stats['median'].to_dict(), brand_stats['median'].to_dict(), global_median)


def _lookup_value(variant, marka, model, bm_median, bm_size, model_median, brand_median, global_median, k=0):
    bm_key = (marka, model)
    if variant == 'A_current':
        if bm_key in bm_median:
            return bm_median[bm_key], 'brand_model'
        if marka in brand_median:
            return brand_median[marka], 'brand'
        return global_median, 'global'

    if variant == 'B_model_tier':
        if bm_key in bm_median:
            return bm_median[bm_key], 'brand_model'
        if model in model_median:
            return model_median[model], 'model'
        if marka in brand_median:
            return brand_median[marka], 'brand'
        return global_median, 'global'

    if variant.startswith('C_shrink'):
        brand_fallback = brand_median.get(marka, global_median)
        if bm_key in bm_median:
            n = bm_size[bm_key]
            value = (n * bm_median[bm_key] + k * brand_fallback) / (n + k)
            return value, 'brand_model_shrunk'
        return brand_fallback, ('brand' if marka in brand_median else 'global')

    raise ValueError(f'bilinmeyen varyant: {variant}')


def _variant_k(variant):
    if variant == 'C_shrink_k5':
        return 5
    if variant == 'C_shrink_k20':
        return 20
    return 0


def compute_oof_variant(X_full, y_full, variant, n_splits=OOF_N_SPLITS, seed=OOF_SEED):
    """hierarchical_price.compute_oof_feature ile AYNI 5-fold OOF deseni - her
    satirin degeri, kendi fiyati DAHIL OLMADAN, DIGER fold'lardan hesaplanir."""
    k = _variant_k(variant)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X_full)
    values = np.empty(n, dtype=float)
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)

    for fold_train_idx, fold_eval_idx in kf.split(df):
        fold_train = df.iloc[fold_train_idx]
        bm_median, bm_size, model_median, brand_median, global_median = _group_stats(fold_train)

        fold_eval = df.iloc[fold_eval_idx]
        for pos, marka, model in zip(fold_eval_idx, fold_eval['marka'], fold_eval['model']):
            val, _ = _lookup_value(variant, marka, model, bm_median, bm_size, model_median,
                                    brand_median, global_median, k=k)
            values[pos] = val
    return values


def build_variant_lookup(X_full, y_full, variant):
    """Inference icin TUM (fold'suz) train verisinden kurulan lookup - hierarchical_price.
    build_price_lookup ile ayni prensip: holdout/val/test bu hesaba HIC girmez."""
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)
    bm_median, bm_size, model_median, brand_median, global_median = _group_stats(df)
    return {
        'variant': variant, 'k': _variant_k(variant),
        'bm_median': bm_median, 'bm_size': bm_size, 'model_median': model_median,
        'brand_median': brand_median, 'global_median': global_median,
    }


def attach_variant_feature(X, lookup):
    variant, k = lookup['variant'], lookup['k']
    values = np.empty(len(X), dtype=float)
    for i, (marka, model) in enumerate(zip(X['marka'], X['model'])):
        val, _ = _lookup_value(variant, marka, model, lookup['bm_median'], lookup['bm_size'],
                               lookup['model_median'], lookup['brand_median'], lookup['global_median'], k=k)
        values[i] = val
    return X.assign(brand_model_median_price_variant=values)


def lookup_variant(marka, model, lookup):
    """hierarchical_price.lookup_price ile ayni sekil - (deger, kaynak_katman) dondurur.
    Ablation raporundaki "hangi katman ne siklikta kullanildi" teshis tablosu icin."""
    variant, k = lookup['variant'], lookup['k']
    return _lookup_value(variant, marka, model, lookup['bm_median'], lookup['bm_size'],
                         lookup['model_median'], lookup['brand_median'], lookup['global_median'], k=k)
