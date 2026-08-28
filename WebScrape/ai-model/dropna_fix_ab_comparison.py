"""Faz 30: preprocess.py'nin dropna/missing-value duzeltmesinin (bkz. preprocess.py
Faz 30 notu) izole A/B karsilastirmasi. Uretim artefaktini (models/lightgbm_final.joblib)
DEGISTIRMEZ, train_dataset.csv'ye DOKUNMAZ, sentetik veri URETMEZ.

DENEY TASARIMI: q99_and_hp_retrain_comparison.py (Faz 29) ile AYNI felsefe - TEK
sabit test seti, TEK izole edilen degisken. Burada izole edilen degisken: kategorik
alanlarin (paket/kasa_turu/renk/yakit_turu/vites) NaN'de satiri dusurmesi (BASELINE,
mevcut production) vs 'Belirtilmemiş'e cevrilmesi + motor_hacmi/motor_gucu NaN'de
satiri dusurmesi (BASELINE) vs NaN birakilmasi (NEW). marka-ici q99/km filtresi VE
hierarchical_price mimarisi HER IKI tarafta da AYNI (kullanicinin 4. ve 5. kurallari).

Adimlar:
1. Ham veriyi oku, marka-ici q99 + km<=1M filtresini uygula (SHARED, degismiyor).
2. Bu ORTAK evrende TEK train/test split (random_state=42) - boylece BASELINE ve
   NEW AYNI fiziksel satirlar uzerinde train/test'e ayrilir.
3. BASELINE-train: bu train parcasina ESKI blanket dropna uygulanir (satir dusurulur).
   NEW-train: kategorikler 'Belirtilmemiş' ile doldurulur, HICBIR satir dusurulmez
   (motor_hacmi/motor_gucu NaN kalir).
4. Test seti HER IKI taraf icin AYNI satirlardir - BASELINE test icin kategorikler
   NaN BIRAKILIR (eski serve.py'nin bos alanlari nasil gordugu), NEW test icin
   'Belirtilmemiş'e cevrilir (yeni serve.py davranisi) - boylece "bu gercek istek
   her iki sistemde de nasil sonuclanir" adil sekilde olculur.

Calistirma (ai-model/ calisma dizini olarak): python dropna_fix_ab_comparison.py
"""
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import (
    CATEGORICAL_FILLNA_COLS, DROP_COLS, REQUIRED_COLS, TRAIN_PATH, UNKNOWN_CATEGORY_VALUE,
    UNKNOWN_FLAG_COLS, load_clean_train_dataset,
)
from train import BASELINE_PARAMS, CATEGORICAL_COLS, to_category
import hierarchical_price as hp

OLD_WATCH_COLS = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi',
                   'motor_gucu', 'yil', 'kilometre', 'yakit_turu', 'vites']
PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
PROBE_CASES = [
    ('Ferrari', '458', 2013, 25_000),
    ('Lamborghini', 'Huracan', 2016, 30_000),
    ('Rolls-Royce', 'Ghost', 2014, 50_000),
    ('Cadillac', 'Escalade', 2015, 208_400),
    ('Bentley', 'Continental', 2015, 60_000),
]


def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {'n': len(y_true), 'MAE': mae, 'MAPE%': mape, 'R2': r2}


def print_row(label, m):
    if m is None or m['n'] == 0:
        print(f'{label:<32} n=0 (veri yok)')
        return
    r2_txt = f"{m['R2']:.4f}" if not np.isnan(m['R2']) else 'n/a'
    print(f"{label:<32} n={m['n']:<7} MAE={m['MAE']:>12,.0f}  MAPE={m['MAPE%']:>7.1f}%  R2={r2_txt}")


def build_shared_universe():
    """load_clean_train_dataset()'in fiyat q99 + km filtresi kadarini (SHARED,
    degismeyen) tekrarlar - kategorik/dropna asamasina GELMEDEN once bir 'ortak
    evren' dondurur (hem BASELINE hem NEW buradan turer)."""
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    brand_q99 = df.groupby('marka')['fiyat'].transform(lambda s: s.quantile(0.99))
    df = df[df['fiyat'] <= brand_q99]
    df = df[df['kilometre'] <= 1_000_000]
    df = df.drop(columns=DROP_COLS)

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)
    df['agir_hasarli'] = df['agir_hasarli'].fillna(0)

    from preprocess import CURRENT_YEAR
    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)
    # bu asamada marka/model/yil/kilometre/fiyat ZATEN hicbir zaman NaN degil
    # (bkz. preprocess.py Faz 30 notu / dogrulanmis calisma) - REQUIRED_COLS
    # icin ek dropna gerekmiyor, sadece guvenlik icin uygulanir.
    return df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)


def prep_baseline(df_subset):
    df = df_subset.copy()
    return df.dropna(subset=[c for c in OLD_WATCH_COLS if c in df.columns])


def prep_new(df_subset):
    df = df_subset.copy()
    for col in CATEGORICAL_FILLNA_COLS:
        df[col] = df[col].fillna(UNKNOWN_CATEGORY_VALUE)
    return df


def to_cat(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category').cat.set_categories(X_train[col].cat.categories)
    return X_train, X_test


def main():
    print('=== ortak evren hazirlaniyor (marka-ici q99 + km filtresi, SHARED) ===')
    universe = build_shared_universe()
    print(f'ortak evren: {len(universe)} satir')

    train_u, test_u = train_test_split(universe, test_size=0.2, random_state=42)
    print(f'train partisyonu (ortak): {len(train_u)}, test partisyonu (HER IKI icin AYNI): {len(test_u)}')

    baseline_train_df = prep_baseline(train_u)
    new_train_df = prep_new(train_u)
    print(f'BASELINE-train (blanket dropna sonrasi): {len(baseline_train_df)}')
    print(f'NEW-train (hicbir satir dusurulmedi): {len(new_train_df)}')

    y_train_baseline = baseline_train_df['fiyat']
    X_train_baseline = baseline_train_df.drop(columns=['fiyat', 'ilan_id'])
    y_train_new = new_train_df['fiyat']
    X_train_new = new_train_df.drop(columns=['fiyat', 'ilan_id'])

    y_test = test_u['fiyat']
    # test kolonlarindaki ORIJINAL eksiklik maskeleri (dolgudan ONCE) - segment
    # analizinde "missing motor_hacmi/motor_gucu/paket olan satirlar" icin.
    test_missing_motor_hacmi = test_u['motor_hacmi'].isna().values
    test_missing_motor_gucu = test_u['motor_gucu'].isna().values
    test_missing_paket = test_u['paket'].isna().values

    X_test_baseline = test_u.drop(columns=['fiyat', 'ilan_id'])  # kategorikler NaN KALIR
    X_test_new = prep_new(test_u).drop(columns=['fiyat', 'ilan_id'])  # kategorikler Belirtilmemiş

    X_train_baseline_c, X_test_baseline_c = to_cat(X_train_baseline, X_test_baseline)
    X_train_new_c, X_test_new_c = to_cat(X_train_new, X_test_new)

    print('\n=== hierarchical_price ekleniyor (mimari DEGISMEDI, sadece girdi verisi farkli) ===')
    X_train_baseline_hp, _ = hp.attach_oof_feature(X_train_baseline_c, y_train_baseline)
    lookup_baseline = hp.build_price_lookup(X_train_baseline_c, y_train_baseline)
    X_test_baseline_hp = hp.attach_lookup_feature(X_test_baseline_c, lookup_baseline)

    X_train_new_hp, _ = hp.attach_oof_feature(X_train_new_c, y_train_new)
    lookup_new = hp.build_price_lookup(X_train_new_c, y_train_new)
    X_test_new_hp = hp.attach_lookup_feature(X_test_new_c, lookup_new)

    print('\n=== iki model egitiliyor (ayni hiperparametreler) ===')
    model_baseline = LGBMRegressor(**BASELINE_PARAMS)
    model_baseline.fit(X_train_baseline_hp, y_train_baseline)
    model_new = LGBMRegressor(**BASELINE_PARAMS)
    model_new.fit(X_train_new_hp, y_train_new)

    pred_baseline = model_baseline.predict(X_test_baseline_hp)
    pred_new = model_new.predict(X_test_new_hp)
    y_test_arr = y_test.values

    marka_test = test_u['marka'].astype(str).values
    model_test = test_u['model'].astype(str).values
    support_counts = X_train_new.groupby(['marka', 'model'], observed=True).size()
    support = np.array([int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)])

    print(f'\n=== OVERALL + SEGMENT KARSILASTIRMASI (n={len(y_test_arr)}) ===')
    mask_5m = y_test_arr > 5_000_000
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)

    segments = [
        ('overall', np.ones(len(y_test_arr), dtype=bool)),
        ('>5.000.000 TL', mask_5m),
        ('premium/luxury (4 marka)', mask_premium),
        ('support n<3', (support >= 0) & (support < 3)),
        ('support 3-9', (support >= 3) & (support < 10)),
        ('support 10-49', (support >= 10) & (support < 50)),
        ('support 50+', support >= 50),
        ('missing motor_hacmi', test_missing_motor_hacmi),
        ('missing motor_gucu', test_missing_motor_gucu),
        ('missing paket', test_missing_paket),
    ]
    print('\n--- BASELINE (mevcut production - blanket dropna) ---')
    for label, mask in segments:
        print_row(label, metrics(y_test_arr[mask], pred_baseline[mask]))
    print('\n--- NEW (dropna/missing-value duzeltmesi) ---')
    for label, mask in segments:
        print_row(label, metrics(y_test_arr[mask], pred_new[mask]))

    print('\n=== TAM (100%) VERIYLE raw/train sayilari + HP support/fallback (probe icin) ===')
    baseline_full = universe.dropna(subset=[c for c in OLD_WATCH_COLS if c in universe.columns])
    new_full = prep_new(universe)
    raw_all = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw_valid = raw_all.dropna(subset=['marka', 'model'])

    y_full_baseline = baseline_full['fiyat']
    X_full_baseline = baseline_full.drop(columns=['fiyat', 'ilan_id'])
    y_full_new = new_full['fiyat']
    X_full_new = new_full.drop(columns=['fiyat', 'ilan_id'])
    for c in CATEGORICAL_COLS:
        X_full_baseline[c] = X_full_baseline[c].astype('category')
        X_full_new[c] = X_full_new[c].astype('category')

    lookup_full_baseline = hp.build_price_lookup(X_full_baseline, y_full_baseline)
    lookup_full_new = hp.build_price_lookup(X_full_new, y_full_new)

    total_recovered_rows = len(new_full) - len(baseline_full)
    baseline_groups = set(zip(X_full_baseline['marka'].astype(str), X_full_baseline['model'].astype(str)))
    new_groups = set(zip(X_full_new['marka'].astype(str), X_full_new['model'].astype(str)))
    recovered_groups = new_groups - baseline_groups
    print(f'toplam geri kazanilan gercek satir (100% veri): {total_recovered_rows}')
    print(f'toplam geri kazanilan brand_model grubu (0 -> >0): {len(recovered_groups)}')

    baseline_counts = X_full_baseline.groupby(['marka', 'model'], observed=True).size()
    new_counts = X_full_new.groupby(['marka', 'model'], observed=True).size()
    all_keys = set(new_counts.index) | set(baseline_counts.index)
    delta_rows = []
    for key in all_keys:
        b = int(baseline_counts.get(key, 0))
        n = int(new_counts.get(key, 0))
        if n > b:
            delta_rows.append({'marka': key[0], 'model': key[1], 'baseline_train_count': b, 'new_train_count': n, 'delta': n - b})
    delta_df = pd.DataFrame(delta_rows).sort_values('delta', ascending=False)
    delta_df.to_csv('reports/dropna_fix_train_count_deltas.csv', index=False, encoding='utf-8-sig')
    print(f'\ntrain_real_count artisi olan grup sayisi: {len(delta_df)} (tam liste reports/dropna_fix_train_count_deltas.csv)')
    print('\nilk 50 (en cok satir kazanan brand_model):')
    print(delta_df.head(50).to_string(index=False))

    print('\n=== PROBE ARACLAR (eski/yeni raw/train/HP support/fallback + final tahmin) ===')
    from preprocess import PRICE_REFERENCE_DATE
    reference_year = PRICE_REFERENCE_DATE.year
    for marka, model, yil, km in PROBE_CASES:
        raw_count = int(len(raw_valid[(raw_valid['marka'] == marka) & (raw_valid['model'] == model)]))
        b_train = int(len(baseline_full[(baseline_full['marka'] == marka) & (baseline_full['model'] == model)]))
        n_train = int(len(new_full[(new_full['marka'] == marka) & (new_full['model'] == model)]))

        yas = max(reference_year - yil, 0)
        b_val, b_src, b_support = hp.lookup_price(marka, model, yas, lookup_full_baseline)
        n_val, n_src, n_support = hp.lookup_price(marka, model, yas, lookup_full_new)
        km_yil = km / (yas if yas > 0 else 1)
        base_row = {
            'marka': marka, 'model': model, 'kasa_turu': 'SUV', 'renk': 'Siyah',
            'motor_hacmi': 6000.0, 'motor_gucu': 400.0, 'yil': yil, 'kilometre': km,
            'yakit_turu': 'Benzin', 'vites': 'Otomatik', 'degisen_sayisi': 0, 'boyali_sayisi': 0,
            'agir_hasarli': 0, 'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0,
            'yas': yas, 'km_yil': km_yil,
        }
        row_baseline = pd.DataFrame([{**base_row, 'paket': np.nan}])
        row_new = pd.DataFrame([{**base_row, 'paket': UNKNOWN_CATEGORY_VALUE}])

        for c in CATEGORICAL_COLS:
            row_baseline[c] = row_baseline[c].astype('category').cat.set_categories(X_train_baseline_hp[c].cat.categories)
            row_new[c] = row_new[c].astype('category').cat.set_categories(X_train_new_hp[c].cat.categories)
        row_baseline = row_baseline.reindex(columns=X_train_baseline_hp.columns)
        row_baseline[hp.FEATURE_COLUMN] = b_val
        row_new = row_new.reindex(columns=X_train_new_hp.columns)
        row_new[hp.FEATURE_COLUMN] = n_val

        pred_b = float(model_baseline.predict(row_baseline)[0])
        pred_n = float(model_new.predict(row_new)[0])

        print(f'\n{marka} {model} ({yil}, {km:,} km, yas={yas}):')
        print(f'  raw={raw_count}')
        print(f'  BASELINE -> train={b_train}  hp_support={b_support} ({b_src})  hp_val={b_val:,.0f}  final_tahmin={pred_b:,.0f}')
        print(f'  NEW      -> train={n_train}  hp_support={n_support} ({n_src})  hp_val={n_val:,.0f}  final_tahmin={pred_n:,.0f}')


if __name__ == '__main__':
    main()
