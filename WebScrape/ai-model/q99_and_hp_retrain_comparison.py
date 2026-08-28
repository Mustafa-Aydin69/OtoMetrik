"""Faz 29: preprocess.py'nin marka-ici q99 kirpmasi + hierarchical_price.py'nin
yas-farkindalikli Theil-Sen egrisi degisikliklerini ESKI (Faz 20-28) pipeline'a
karsi AYNI holdout uzerinde karsilastiran tek-seferlik rapor scripti. Uretim
artefaktini (models/lightgbm_final.joblib) DEGISTIRMEZ - sadece stdout'a rapor basar.

Deney tasarimi (adil A/B icin): NEW (marka-ici q99) temizlenmis tam veri 80/20
bolunur (random_state=42) - bu TEK bolme hem OLD hem NEW icin train/test iskeletini
sabitler. Test seti (X_test) HER IKI taraf icin BIREBIR AYNI satirlardir (premium
markalari icerir, cunku NEW temizlemeden geldi). OLD-train, bu ayni train
partisyonuna GERIYE DONUK olarak eski GLOBAL q99 esigi uygulanarak turetilir
(train_dataset.csv'yi bastan okuyup ayri bir bolme yapmak farkli satirlar
verir ve karsilastirmayi GECERSIZ kilardi). Boylece izole edilen TEK degisken:
(a) satir dahil etme stratejisi (global vs marka-ici q99) ve (b) hierarchical_price
hesaplama yontemi (duz medyan vs yas-farkindalikli Theil-Sen) - split, hiperparametreler
ve test seti SABIT.

Calistirma (ai-model/ calisma dizini olarak): python q99_and_hp_retrain_comparison.py
"""
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import load_clean_train_dataset
from train import BASELINE_PARAMS, CATEGORICAL_COLS, to_category
import hierarchical_price as new_hp

REPO_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()


def _load_old_hierarchical_price_module():
    """git HEAD'deki (Faz 29 oncesi) hierarchical_price.py'yi ayri bir modul
    olarak yukler - duz-medyan bazli ESKI davranisi, dosyayi elle yeniden
    yazmadan, gercek eski koddan calistirmak icin."""
    src = subprocess.check_output(
        ['git', 'show', 'HEAD:WebScrape/ai-model/hierarchical_price.py'],
        cwd=REPO_ROOT, text=True,
    )
    mod = types.ModuleType('old_hierarchical_price')
    exec(compile(src, 'old_hierarchical_price(HEAD)', 'exec'), mod.__dict__)
    return mod


old_hp = _load_old_hierarchical_price_module()

PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
PROBE_CASES = [
    ('Cadillac', 'Escalade', 2015, 208_400),
    ('Ferrari', '458', 2013, 25_000),
    ('Lamborghini', 'Huracan', 2016, 30_000),
    ('Bentley', 'Continental', 2015, 60_000),
    ('Rolls-Royce', 'Ghost', 2014, 50_000),
]


def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {'n': len(y_true), 'MAE': mae, 'MAPE%': mape, 'R2': r2}


def print_row(label, m):
    if m is None:
        print(f'{label:<28} n=0 (veri yok)')
        return
    r2_txt = f"{m['R2']:.4f}" if not np.isnan(m['R2']) else 'n/a'
    print(f"{label:<28} n={m['n']:<6} MAE={m['MAE']:>12,.0f}  MAPE={m['MAPE%']:>7.1f}%  R2={r2_txt}")


def main():
    print('=== veri hazirlama (NEW: marka-ici q99 temizlenmis tam veri) ===')
    df = load_clean_train_dataset()
    y_full = df['fiyat']
    X_full = df.drop(columns=['fiyat', 'ilan_id'])
    for c in CATEGORICAL_COLS:
        X_full[c] = X_full[c].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    print(f'train partisyonu: {len(X_train)}, test partisyonu (HER IKI model icin AYNI): {len(X_test)}')

    # OLD-train: ayni train partisyonuna GERIYE DONUK global q99 uygulanir.
    old_threshold = y_train.quantile(0.99)
    old_mask = y_train <= old_threshold
    X_train_old_raw, y_train_old = X_train[old_mask.values], y_train[old_mask.values]
    print(f'OLD-train (geriye donuk global q99, esik={old_threshold:,.0f}): {len(X_train_old_raw)} satir '
          f'({len(X_train) - len(X_train_old_raw)} satir silindi)')
    print(f'NEW-train (marka-ici q99, ek filtre YOK): {len(X_train)} satir')

    # kategori setleri her varyant kendi train'ine gore ayri hizalanir (OLD icin
    # premium markalar 'gorulmemis kategori' olarak native missing'e duser -
    # gercek eski model davranisini BIREBIR simule eder).
    X_train_old, X_test_old = to_category(X_train_old_raw, X_test)
    X_train_new, X_test_new = to_category(X_train, X_test)

    print('\n=== hierarchical_price ozelligi ekleniyor (OOF egitim, tam-veri inference lookup) ===')
    X_train_old_hp, _ = old_hp.attach_oof_feature(X_train_old, y_train_old)
    old_lookup = old_hp.build_price_lookup(X_train_old, y_train_old)
    X_test_old_hp = old_hp.attach_lookup_feature(X_test_old, old_lookup)

    X_train_new_hp, _ = new_hp.attach_oof_feature(X_train_new, y_train)
    new_lookup = new_hp.build_price_lookup(X_train_new, y_train)
    X_test_new_hp = new_hp.attach_lookup_feature(X_test_new, new_lookup)

    print('\n=== iki model egitiliyor (ayni hiperparametreler, train.BASELINE_PARAMS) ===')
    model_old = LGBMRegressor(**BASELINE_PARAMS)
    model_old.fit(X_train_old_hp, y_train_old)
    model_new = LGBMRegressor(**BASELINE_PARAMS)
    model_new.fit(X_train_new_hp, y_train)

    pred_old = model_old.predict(X_test_old_hp)
    pred_new = model_new.predict(X_test_new_hp)

    marka_test = X_test['marka'].astype(str).values
    model_test = X_test['model'].astype(str).values
    y_test_arr = y_test.values

    # support bucket: her test satirinin OWN marka+model grubunun NEW-train'deki
    # satir sayisi (rarity gostergesi - OLD-train icin degil, cunku o zaten
    # premium/rare olani sistematik siliyor, karsilastirma NEW-train destegine
    # gore yapilmali).
    support_counts = X_train_new.groupby(['marka', 'model'], observed=True).size()
    support = np.array([
        int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)
    ])

    print('\n=== OVERALL + SEGMENT KARSILASTIRMASI (ayni test seti, n=%d) ===' % len(y_test_arr))
    print('\n--- OLD (global q99 + duz medyan hierarchical_price) ---')
    print_row('overall', metrics(y_test_arr, pred_old))
    mask_5m = y_test_arr > 5_000_000
    print_row('>5.000.000 TL', metrics(y_test_arr[mask_5m], pred_old[mask_5m]))
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)
    print_row('premium/luxury (4 marka)', metrics(y_test_arr[mask_premium], pred_old[mask_premium]))
    for lo, hi, label in [(0, 3, 'support n<3'), (3, 10, 'support 3-9'), (10, 50, 'support 10-49'), (50, 10**9, 'support 50+')]:
        m = (support >= lo) & (support < hi)
        print_row(f'  bucket: {label}', metrics(y_test_arr[m], pred_old[m]))

    print('\n--- NEW (marka-ici q99 + yas-farkindalikli Theil-Sen hierarchical_price) ---')
    print_row('overall', metrics(y_test_arr, pred_new))
    print_row('>5.000.000 TL', metrics(y_test_arr[mask_5m], pred_new[mask_5m]))
    print_row('premium/luxury (4 marka)', metrics(y_test_arr[mask_premium], pred_new[mask_premium]))
    for lo, hi, label in [(0, 3, 'support n<3'), (3, 10, 'support 3-9'), (10, 50, 'support 10-49'), (50, 10**9, 'support 50+')]:
        m = (support >= lo) & (support < hi)
        print_row(f'  bucket: {label}', metrics(y_test_arr[m], pred_new[m]))

    print('\n=== PROBE ORNEKLER (eski/yeni final tahmin, hierarchical referans, fallback seviyesi, destek n) ===')
    from preprocess import PRICE_REFERENCE_DATE
    reference_year = PRICE_REFERENCE_DATE.year

    for marka, model, yil, km in PROBE_CASES:
        yas = max(reference_year - yil, 0)
        km_yil = km / (yas if yas > 0 else 1)
        row = pd.DataFrame([{
            'marka': marka, 'model': model, 'paket': np.nan, 'kasa_turu': 'SUV', 'renk': 'Siyah',
            'motor_hacmi': 6000.0, 'motor_gucu': 400.0, 'yil': yil, 'kilometre': km,
            'yakit_turu': 'Benzin', 'vites': 'Otomatik', 'degisen_sayisi': 0, 'boyali_sayisi': 0,
            'agir_hasarli': 0, 'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0,
            'yas': yas, 'km_yil': km_yil,
        }])

        old_hp_val, old_src = old_hp.lookup_price(marka, model, old_lookup)
        old_n = {
            'brand_model': int(((X_train_old_raw['marka'] == marka) & (X_train_old_raw['model'] == model)).sum()),
            'model': int((X_train_old_raw['model'] == model).sum()),
            'brand': int((X_train_old_raw['marka'] == marka).sum()),
            'global': len(X_train_old_raw),
        }[old_src]
        new_hp_val, new_src, new_n = new_hp.lookup_price(marka, model, yas, new_lookup)

        row_old = row.copy()
        row_old[old_hp.FEATURE_COLUMN] = old_hp_val
        row_old_c = row_old.reindex(columns=X_train_old_hp.columns)
        for c in CATEGORICAL_COLS:
            row_old_c[c] = row_old_c[c].astype('category').cat.set_categories(X_train_old_hp[c].cat.categories)
        final_old = float(model_old.predict(row_old_c)[0])

        row_new = row.copy()
        row_new[new_hp.FEATURE_COLUMN] = new_hp_val
        row_new_c = row_new.reindex(columns=X_train_new_hp.columns)
        for c in CATEGORICAL_COLS:
            row_new_c[c] = row_new_c[c].astype('category').cat.set_categories(X_train_new_hp[c].cat.categories)
        final_new = float(model_new.predict(row_new_c)[0])

        print(f'\n{marka} {model} ({yil}, {km:,} km, yas={yas}):')
        print(f'  OLD -> final tahmin={final_old:,.0f}  hp_ref={old_hp_val:,.0f}  fallback={old_src}  destek_n={old_n}')
        print(f'  NEW -> final tahmin={final_new:,.0f}  hp_ref={new_hp_val:,.0f}  fallback={new_src}  destek_n={new_n}')


if __name__ == '__main__':
    main()
