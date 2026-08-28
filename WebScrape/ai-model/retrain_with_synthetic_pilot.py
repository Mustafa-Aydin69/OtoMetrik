"""Faz 30 pilot v3 - PRODUCTION RETRAIN: gercek egitim verisi + 18 satirlik
sabit pilot sentetik veri (data/output/synthetic_pilot.csv, v3 - v1_reference
DEGIL) ile models/lightgbm_final.joblib'i GUNCELLER.

KAPSAM: SADECE Ferrari 458 / Lamborghini Huracan / Rolls-Royce Ghost. Baska
hicbir brand_model'e sentetik eklenmedi (18 satirlik sabit dosya, YENIDEN
URETILMEDI). 2224 aday gruba GECILMEDI.

MIMARI KURALLARI (kullanicinin talebiyle BIREBIR):
- hierarchical_price OOF + full lookup SADECE gercek training verisinden
  (X_real, y_real) kurulur - sentetik satirlar bu hesaba HIC girmez, sadece
  SONUCTAN (attach_lookup_feature) deger okur.
- hp_support (motor_gucu peer-confidence ozeti, serve.py'nin /predict
  confidence'inda kullandigi) SADECE gercek X_real'den kurulur - sentetik
  satirlar support sayisini ARTIRMAZ (bkz. dogrulama adimi asagida).
- category_sets/feature_columns gercek egitim verisinin kategori
  cesitliliginden turetilir (sentetik satirlar zaten sadece gercek
  parent'lardan KOPYALANMIS kategoriler tasidigi icin yeni kategori
  EKLEMEZ).
- SADECE nihai LightGBM model.fit() cagrisi gercek+sentetik BIRLESIK
  matristir - sample_weight: gercek=1.0, sentetik=0.50 (pilot A/B/8-seed
  testinde onaylanan agirlik).
- train_dataset.csv'ye HICBIR SEKILDE yazilmaz, synthetic_pilot.csv AYRI
  dosya olarak kalir.

Calistirma (ai-model/ calisma dizini olarak): python retrain_with_synthetic_pilot.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import PRICE_REFERENCE_DATE, load_clean_train_dataset
from train import (
    BASELINE_PARAMS, CATEGORICAL_COLS, MODEL_PATH, apply_saved_categories,
    load_cars1_holdout,
)
from hp_support import build_support_summary
import hierarchical_price as hp

SYNTHETIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
SYNTHETIC_WEIGHT = 0.50
SYNTHETIC_VERSION = 'v3'
SYNTHETIC_GENERATOR_VERSION = (
    'generate_synthetic_pilot.py v3 - motor_hacmi/motor_gucu tek-donor kopya (interpole edilmez), '
    'sadece yil/km/fiyat interpole, hierarchical_price curve guard (+-12%), '
    'gurultu maks +-3% (parent sinirina kirpilir)'
)
PILOT_GROUPS = ['Ferrari|458', 'Lamborghini|Huracan', 'Rolls-Royce|Ghost']


def main():
    print('=== gercek egitim verisi (production preprocessing, degismedi) ===')
    clean = load_clean_train_dataset()
    y_real = clean['fiyat']
    X_real = clean.drop(columns=['fiyat', 'ilan_id'])
    for c in CATEGORICAL_COLS:
        X_real[c] = X_real[c].astype('category')
    print(f'gercek egitim satiri: {len(X_real)}')

    print('\n=== sabit pilot sentetik veri okunuyor (v3, YENIDEN URETILMEDI) ===')
    synth = pd.read_csv(SYNTHETIC_PATH)
    assert len(synth) == 18, f'beklenmeyen sentetik satir sayisi: {len(synth)} (18 bekleniyordu)'
    assert set(synth['is_synthetic'].unique()) == {1}, 'is_synthetic=1 olmayan satir var'
    assert set(synth['source'].unique()) == {'synthetic_pilot'}, "source='synthetic_pilot' olmayan satir var"
    print(f'{len(synth)} sentetik satir dogrulandi: {synth.groupby(["marka", "model"]).size().to_dict()}')

    synth_cols = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
                  'yil', 'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli']
    synth_X = synth[synth_cols].copy()
    synth_y = synth['fiyat']
    synth_X['degisen_sayisi_bilinmiyor'] = 0
    synth_X['boyali_sayisi_bilinmiyor'] = 0
    synth_X['yas'] = (PRICE_REFERENCE_DATE.year - synth_X['yil']).clip(lower=0)
    synth_X['km_yil'] = synth_X['kilometre'] / synth_X['yas'].replace(0, 1)
    synth_X = synth_X.reindex(columns=X_real.columns)
    for c in CATEGORICAL_COLS:
        synth_X[c] = synth_X[c].astype('category').cat.set_categories(X_real[c].cat.categories)

    print('\n=== hp_support (motor_gucu confidence ozeti) - SADECE gercek X_real ===')
    hp_support_real = build_support_summary(X_real)
    for key in PILOT_GROUPS:
        entry = hp_support_real['model_stats'].get(key)
        print(f'  {key}: hp_support model_stats count = {entry["count"] if entry else "(yok, marka/global fallback)"}')

    print('\n=== hierarchical_price - SADECE gercek (X_real, y_real) ===')
    X_real_hp, _ = hp.attach_oof_feature(X_real, y_real)
    lookup_real = hp.build_price_lookup(X_real, y_real)
    synth_X_hp = hp.attach_lookup_feature(synth_X, lookup_real)
    for key in PILOT_GROUPS:
        marka, model = key.split('|')
        c = lookup_real['brand_model_curve'].get(f'{marka}\x1f{model}')
        print(f'  {key}: brand_model_curve n = {c["n"] if c else "(yok)"}')

    print('\n=== NIHAI LightGBM egitimi: gercek(w=1.0) + sentetik(w=%.2f) ===' % SYNTHETIC_WEIGHT)
    X_combined = pd.concat([X_real_hp, synth_X_hp], ignore_index=True)
    y_combined = pd.concat([y_real.reset_index(drop=True), synth_y.reset_index(drop=True)], ignore_index=True)
    w_combined = pd.concat([
        pd.Series(1.0, index=range(len(X_real_hp))),
        pd.Series(SYNTHETIC_WEIGHT, index=range(len(synth_X_hp))),
    ], ignore_index=True)
    print(f'toplam egitim satiri (model.fit girdisi): {len(X_combined)} '
          f'(gercek={len(X_real_hp)} + sentetik={len(synth_X_hp)})')

    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(X_combined, y_combined, sample_weight=w_combined)

    print('\n=== dis holdout (cars1, hic gorulmemis) ===')
    holdout_df = load_cars1_holdout()
    y_holdout = holdout_df['fiyat']
    X_holdout = holdout_df.drop(columns=['fiyat', 'ilan_id']).reindex(columns=X_real.columns)
    for c in CATEGORICAL_COLS:
        X_holdout[c] = X_holdout[c].astype('category').cat.set_categories(X_real[c].cat.categories)
    X_holdout_hp = hp.attach_lookup_feature(X_holdout, lookup_real)
    pred_holdout = model.predict(X_holdout_hp)
    mae = mean_absolute_error(y_holdout, pred_holdout)
    rmse = float(np.sqrt(mean_squared_error(y_holdout, pred_holdout)))
    r2 = r2_score(y_holdout, pred_holdout)
    print(f'MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}  (n={len(y_holdout)})')

    generated_at = synth['generated_at'].iloc[0]
    synthetic_seed = int(synth['synthetic_seed'].iloc[0])

    artifact = {
        'model': model,
        'categorical_cols': CATEGORICAL_COLS,
        'category_sets': {col: X_real[col].cat.categories for col in CATEGORICAL_COLS},
        'feature_columns': list(X_combined.columns),
        'reference_year': PRICE_REFERENCE_DATE.year,
        'hp_support': hp_support_real,
        'hierarchical_price': lookup_real,
        'synthetic_enabled': True,
        'synthetic_version': SYNTHETIC_VERSION,
        'synthetic_row_count': int(len(synth)),
        'synthetic_weight': SYNTHETIC_WEIGHT,
        'synthetic_groups': PILOT_GROUPS,
        'synthetic_generator_version': SYNTHETIC_GENERATOR_VERSION,
        'synthetic_seed': synthetic_seed,
        'synthetic_generated_at': generated_at,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f'\nmodel kaydedildi: {MODEL_PATH}')

    print('\n=== SMOKE TEST ===')
    sample = X_holdout_hp.iloc[[0]]
    sample_aligned = apply_saved_categories(sample, artifact)
    pred = artifact['model'].predict(sample_aligned)[0]
    actual = y_holdout.iloc[0]
    print(f'smoke-test -> tahmin: {pred:,.0f} (gercek fiyat: {actual:,.0f})')


if __name__ == '__main__':
    main()
