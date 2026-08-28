"""Faz 31 - PRODUCTION RETRAIN: gercek egitim verisi + Faz30 pilot (18 satir)
+ Faz31 second-wave (38 satir) ile models/lightgbm_final.joblib'i GUNCELLER.
Ikisi de sample_weight=0.50. Audi RS DAHIL DEGIL (Faz31 planinda hic yok).

MIMARI (retrain_with_synthetic_pilot.py ile AYNI ilke, iki sentetik dalgaya
genellenmis): hierarchical_price OOF + full lookup + hp_support SADECE gercek
(X_real, y_real) - iki sentetik dalga da bu hesaba HIC girmez, sadece
SONUCTAN (attach_lookup_feature) deger okur. Nihai LightGBM fit'i gercek+
wave30+wave31 BIRLESIK matris (sample_weight: gercek=1.0, HER IKI dalga=0.50).

Calistirma (ai-model/ calisma dizini olarak): python retrain_with_wave31.py
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
from train import BASELINE_PARAMS, CATEGORICAL_COLS, MODEL_PATH, apply_saved_categories, load_cars1_holdout
from hp_support import build_support_summary
import hierarchical_price as hp

WAVE30_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
WAVE31_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_second_wave_preview.csv')
SYNTHETIC_WEIGHT = 0.50

WAVE30_GROUPS = ['Ferrari|458', 'Lamborghini|Huracan', 'Rolls-Royce|Ghost']
WAVE31_GROUPS = ['Dodge|Ram', 'Rolls-Royce|Wraith', 'Lexus|LS', 'Aston Martin|Vantage',
                 'Bentley|Flying Spur', 'Mercedes - Benz|Maybach S', 'Mercedes - Benz|V Serisi']

WAVE31_GENERATOR_VERSION = (
    'generate_second_wave_preview.py - Faz30 mimarisi (motor_hacmi/motor_gucu tek-donor, '
    'sadece yil/km/fiyat interpole, curve guard +-12%, gurultu +-3%) + nesil-kume kisitlamasi '
    '(Dodge Ram: 2004 legacy haric, sadece modern SUV kumesi; Lexus LS: 2015 eski nesil haric; '
    'Bentley Flying Spur: gen2/gen3 capraz cift yasak; Audi RS: heterojen model etiketi nedeniyle '
    'tamamen haric tutuldu, bu dalgada YOK)'
)


def prep_synth(raw, ref_cols):
    cols = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
            'yil', 'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli']
    X = raw[cols].copy()
    y = raw['fiyat'].reset_index(drop=True)
    X['degisen_sayisi_bilinmiyor'] = 0
    X['boyali_sayisi_bilinmiyor'] = 0
    X['yas'] = (PRICE_REFERENCE_DATE.year - X['yil']).clip(lower=0)
    X['km_yil'] = X['kilometre'] / X['yas'].replace(0, 1)
    X = X.reindex(columns=ref_cols)
    return X, y


def main():
    print('=== gercek egitim verisi (production preprocessing, degismedi) ===')
    clean = load_clean_train_dataset()
    y_real = clean['fiyat']
    X_real = clean.drop(columns=['fiyat', 'ilan_id'])
    for c in CATEGORICAL_COLS:
        X_real[c] = X_real[c].astype('category')
    print(f'gercek egitim satiri: {len(X_real)}')

    print('\n=== sentetik dalgalar okunuyor (YENIDEN URETILMEDI) ===')
    wave30 = pd.read_csv(WAVE30_PATH)
    wave31 = pd.read_csv(WAVE31_PATH)
    assert len(wave30) == 18, f'wave30 beklenmeyen satir sayisi: {len(wave30)}'
    assert len(wave31) == 38, f'wave31 beklenmeyen satir sayisi: {len(wave31)}'
    assert len(wave31[(wave31['marka'] == 'Audi') & (wave31['model'] == 'RS')]) == 0, 'Audi RS wave31 icinde bulundu!'
    print(f'wave30: {len(wave30)} ({wave30.groupby(["marka", "model"]).size().to_dict()})')
    print(f'wave31: {len(wave31)} ({wave31.groupby(["marka", "model"]).size().to_dict()})')

    wave30_X, wave30_y = prep_synth(wave30, X_real.columns)
    wave31_X, wave31_y = prep_synth(wave31, X_real.columns)
    for c in CATEGORICAL_COLS:
        wave30_X[c] = wave30_X[c].astype('category').cat.set_categories(X_real[c].cat.categories)
        wave31_X[c] = wave31_X[c].astype('category').cat.set_categories(X_real[c].cat.categories)

    print('\n=== hp_support (motor_gucu confidence ozeti) - SADECE gercek X_real ===')
    hp_support_real = build_support_summary(X_real)

    print('\n=== hierarchical_price - SADECE gercek (X_real, y_real) ===')
    X_real_hp, _ = hp.attach_oof_feature(X_real, y_real)
    lookup_real = hp.build_price_lookup(X_real, y_real)
    wave30_hp = hp.attach_lookup_feature(wave30_X, lookup_real)
    wave31_hp = hp.attach_lookup_feature(wave31_X, lookup_real)

    print('\n=== dogrulama: 10 grubun hp_support/hierarchical_price GERCEK sayilari ===')
    for key in WAVE30_GROUPS + WAVE31_GROUPS:
        marka, model = key.split('|')
        c = lookup_real['brand_model_curve'].get(f'{marka}\x1f{model}')
        s = hp_support_real['model_stats'].get(f'{marka}\x1f{model}')
        print(f'  {key}: hierarchical_price n={c["n"] if c else None}  hp_support count={s["count"] if s else None}')

    print('\n=== NIHAI LightGBM egitimi: gercek(w=1.0) + wave30(w=%.2f) + wave31(w=%.2f) ===' % (SYNTHETIC_WEIGHT, SYNTHETIC_WEIGHT))
    X_combined = pd.concat([X_real_hp, wave30_hp, wave31_hp], ignore_index=True)
    y_combined = pd.concat([y_real.reset_index(drop=True), wave30_y, wave31_y], ignore_index=True)
    w_combined = pd.concat([
        pd.Series(1.0, index=range(len(X_real_hp))),
        pd.Series(SYNTHETIC_WEIGHT, index=range(len(wave30_hp))),
        pd.Series(SYNTHETIC_WEIGHT, index=range(len(wave31_hp))),
    ], ignore_index=True)
    total_synth = len(wave30_hp) + len(wave31_hp)
    print(f'toplam egitim satiri: {len(X_combined)} (gercek={len(X_real_hp)} + wave30={len(wave30_hp)} + wave31={len(wave31_hp)} = sentetik toplam {total_synth})')

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

    wave30_generated_at = wave30['generated_at'].iloc[0]
    wave30_seed = int(wave30['synthetic_seed'].iloc[0])
    wave31_generated_at = wave31['generated_at'].iloc[0]
    wave31_seed = int(wave31['synthetic_seed'].iloc[0])

    artifact = {
        'model': model,
        'categorical_cols': CATEGORICAL_COLS,
        'category_sets': {col: X_real[col].cat.categories for col in CATEGORICAL_COLS},
        'feature_columns': list(X_combined.columns),
        'reference_year': PRICE_REFERENCE_DATE.year,
        'hp_support': hp_support_real,
        'hierarchical_price': lookup_real,
        'synthetic_enabled': True,
        'synthetic_weight': SYNTHETIC_WEIGHT,
        'synthetic_total_rows': int(total_synth),
        'synthetic_waves': {
            'wave30': {
                'row_count': int(len(wave30)),
                'groups': WAVE30_GROUPS,
                'generator_version': (
                    'generate_synthetic_pilot.py v3 - motor_hacmi/motor_gucu tek-donor kopya '
                    '(interpole edilmez), sadece yil/km/fiyat interpole, curve guard (+-12%), '
                    'gurultu maks +-3% (parent sinirina kirpilir)'
                ),
                'seed': wave30_seed,
                'generated_at': wave30_generated_at,
            },
            'wave31': {
                'row_count': int(len(wave31)),
                'groups': WAVE31_GROUPS,
                'generator_version': WAVE31_GENERATOR_VERSION,
                'seed': wave31_seed,
                'generated_at': wave31_generated_at,
            },
        },
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
