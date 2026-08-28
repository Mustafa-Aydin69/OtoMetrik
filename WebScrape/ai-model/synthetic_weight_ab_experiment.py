"""Faz 30 pilot v2: Ferrari 458/Huracan/Ghost sentetik verisiyle (bkz.
generate_synthetic_pilot.py -> data/output/synthetic_pilot.csv) 4 modelli
sample_weight ablation'i. Uretim artefaktini DEGISTIRMEZ, train_dataset.csv'ye
DOKUNMAZ.

TASARIM (kullanicinin kurallariyla BIREBIR):
- Gercek veri TEK train/test split'e ayrilir (random_state=42) - test seti
  SENTETIK GORMEZ, hicbir asamada.
- hierarchical_price OOF/full-lookup SADECE gercek train'den kurulur - sentetik
  satirlar bu hesaba HIC girmez, sadece SONUCTAN (attach_lookup_feature) deger
  okur (bkz. hierarchical_price.py mimarisi, degismedi).
- A) baseline: sadece gercek train (sample_weight=1.0 hepsi).
- B/C/D: gercek train + 18 sentetik pilot satiri, gercek weight=1.0, sentetik
  weight=0.25/0.50/1.00 - AYNI hiperparametreler (train.BASELINE_PARAMS,
  random_state dahil).

Calistirma (ai-model/ calisma dizini olarak): python synthetic_weight_ab_experiment.py
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import PRICE_REFERENCE_DATE, UNKNOWN_CATEGORY_VALUE, load_clean_train_dataset
from train import BASELINE_PARAMS, CATEGORICAL_COLS
import hierarchical_price as hp

SYNTHETIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
PILOT_GROUPS = [('Ferrari', '458'), ('Lamborghini', 'Huracan'), ('Rolls-Royce', 'Ghost')]
WEIGHTS = [('B (W=0.25)', 0.25), ('C (W=0.50)', 0.50), ('D (W=1.00)', 1.00)]
PROBE_CASES = [
    ('Ferrari', '458', 2013, 25_000),
    ('Lamborghini', 'Huracan', 2016, 30_000),
    ('Rolls-Royce', 'Ghost', 2014, 50_000),
]


def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {'n': len(y_true), 'MAE': mae, 'RMSE': rmse, 'MAPE%': mape, 'R2': r2}


def print_row(label, m):
    if m is None or m['n'] == 0:
        print(f'{label:<28} n=0 (veri yok)')
        return
    r2_txt = f"{m['R2']:.4f}" if not np.isnan(m['R2']) else 'n/a'
    print(f"{label:<28} n={m['n']:<6} MAE={m['MAE']:>11,.0f}  RMSE={m['RMSE']:>11,.0f}  MAPE={m['MAPE%']:>7.1f}%  R2={r2_txt}")


def to_cat(X_train, *others):
    X_train = X_train.copy()
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
    result = [X_train]
    for X in others:
        X = X.copy()
        for col in CATEGORICAL_COLS:
            X[col] = X[col].astype('category').cat.set_categories(X_train[col].cat.categories)
        result.append(X)
    return result


def main():
    print('=== gercek veri: TEMIZ dataset + split (random_state=42) ===')
    clean = load_clean_train_dataset()
    y_full = clean['fiyat']
    X_full = clean.drop(columns=['fiyat', 'ilan_id'])
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    print(f'gercek train: {len(X_train)}, gercek test (SENTETIK GORMEZ): {len(X_test)}')

    print('\n=== sentetik pilot okunuyor (SADECE OKUMA) ===')
    synth = pd.read_csv(SYNTHETIC_PATH)
    print(f'{len(synth)} sentetik satir: {synth.groupby(["marka", "model"]).size().to_dict()}')
    synth_cols = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
                  'yil', 'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi',
                  'agir_hasarli', 'fiyat']
    synth_X = synth[[c for c in synth_cols if c != 'fiyat']].copy()
    synth_y = synth['fiyat']
    synth_X['degisen_sayisi_bilinmiyor'] = 0
    synth_X['boyali_sayisi_bilinmiyor'] = 0
    synth_X['yas'] = (PRICE_REFERENCE_DATE.year - synth_X['yil']).clip(lower=0)
    synth_X['km_yil'] = synth_X['kilometre'] / synth_X['yas'].replace(0, 1)
    synth_X = synth_X.reindex(columns=X_train.columns)

    X_train_c, X_test_c, synth_X_c = to_cat(X_train, X_test, synth_X)

    print('\n=== hierarchical_price: SADECE gercek train (bkz. modul kurali) ===')
    X_train_hp, _ = hp.attach_oof_feature(X_train_c, y_train)
    lookup_real = hp.build_price_lookup(X_train_c, y_train)
    X_test_hp = hp.attach_lookup_feature(X_test_c, lookup_real)
    synth_X_hp = hp.attach_lookup_feature(synth_X_c, lookup_real)  # sentetik SADECE OKUR, hic etkilemez

    print('\n=== 4 model egitiliyor (A=baseline, B/C/D=sentetik+weight) ===')
    models = {}
    w_real = pd.Series(1.0, index=X_train_hp.index)

    model_a = LGBMRegressor(**BASELINE_PARAMS)
    model_a.fit(X_train_hp, y_train, sample_weight=w_real)
    models['A (baseline, gercek only)'] = model_a

    for label, w in WEIGHTS:
        X_combined = pd.concat([X_train_hp, synth_X_hp], ignore_index=True)
        y_combined = pd.concat([y_train.reset_index(drop=True), synth_y.reset_index(drop=True)], ignore_index=True)
        w_combined = pd.concat([w_real.reset_index(drop=True), pd.Series([w] * len(synth_y))], ignore_index=True)
        model = LGBMRegressor(**BASELINE_PARAMS)
        model.fit(X_combined, y_combined, sample_weight=w_combined)
        models[label] = model

    preds = {label: m.predict(X_test_hp) for label, m in models.items()}
    y_test_arr = y_test.values
    marka_test = X_test['marka'].astype(str).values
    model_test = X_test['model'].astype(str).values
    support_counts = X_train.groupby(['marka', 'model'], observed=True).size()
    support = np.array([int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)])

    print(f'\n=== SEGMENT KARSILASTIRMASI (n={len(y_test_arr)}, ayni test seti) ===')
    mask_5m = y_test_arr > 5_000_000
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)
    segments = [
        ('overall', np.ones(len(y_test_arr), dtype=bool)),
        ('>5.000.000 TL', mask_5m),
        ('premium/luxury (4 marka)', mask_premium),
        ('support <3', support < 3),
        ('support 3-9', (support >= 3) & (support < 10)),
        ('support 10-49', (support >= 10) & (support < 50)),
        ('support 50+', support >= 50),
    ]
    for label in models:
        print(f'\n--- {label} ---')
        for seg_label, mask in segments:
            print_row(seg_label, metrics(y_test_arr[mask], preds[label][mask]))

    print('\n=== PILOT 3 MODEL - GERCEK HOLDOUT SATIR SATIR ===')
    for marka, model in PILOT_GROUPS:
        mask = (marka_test == marka) & (model_test == model)
        n = int(mask.sum())
        print(f'\n{marka} {model}: n_test={n}')
        if n == 0:
            print('  (bu grup icin test setinde gercek satir yok - kucuk grup, %20 split kaymis olabilir)')
            continue
        idxs = np.where(mask)[0]
        for idx in idxs:
            actual = y_test_arr[idx]
            print(f'  actual={actual:,.0f}')
            for label in models:
                p = preds[label][idx]
                ae = abs(p - actual)
                pe = 100 * ae / actual
                print(f'    {label:<28} pred={p:>13,.0f}  abs_err={ae:>12,.0f}  pct_err={pe:6.1f}%')

    print('\n=== PROBE (sabit girdi, 4 model) ===')
    reference_year = PRICE_REFERENCE_DATE.year
    for marka, model, yil, km in PROBE_CASES:
        yas = max(reference_year - yil, 0)
        km_yil = km / (yas if yas > 0 else 1)
        hp_val, hp_src, hp_support = hp.lookup_price(marka, model, yas, lookup_real)
        row = pd.DataFrame([{
            'marka': marka, 'model': model, 'paket': UNKNOWN_CATEGORY_VALUE, 'kasa_turu': 'Coupe' if marka != 'Rolls-Royce' else 'Sedan',
            'renk': 'Siyah', 'motor_hacmi': 5000.0, 'motor_gucu': 570.0, 'yil': yil, 'kilometre': km,
            'yakit_turu': 'Benzin', 'vites': 'Otomatik', 'degisen_sayisi': 0, 'boyali_sayisi': 0,
            'agir_hasarli': 0, 'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0,
            'yas': yas, 'km_yil': km_yil,
        }])
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype('category').cat.set_categories(X_train_hp[c].cat.categories)
        row = row.reindex(columns=X_train_hp.columns)
        row[hp.FEATURE_COLUMN] = hp_val

        print(f'\n{marka} {model} ({yil}, {km:,} km, yas={yas}):')
        print(f'  hp_reference={hp_val:,.0f}  hp_support={hp_support}  fallback={hp_src}')
        for label, m in models.items():
            pred = float(m.predict(row)[0])
            print(f'  {label:<28} pred={pred:,.0f}')


if __name__ == '__main__':
    main()
