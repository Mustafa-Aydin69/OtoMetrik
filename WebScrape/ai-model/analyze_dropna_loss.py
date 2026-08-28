"""Faz 30 devam: load_clean_train_dataset() icindeki kayip asamalarini
(brand-q99 fiyat filtresi -> km filtresi -> final dropna()) SATIR BAZINDA
izleyen, SADECE ANALIZ scripti. Kod/veri/artefakt DEGISTIRMEZ.

Onceki analizde (analyze_synthetic_data_candidates.py) bulunan 256 marka+model
grubu (raw_real_count>0, train_real_count=0) icin: HANGI asamada, HANGI
kolon(lar) yuzunden kaybolduklarini kolon bazinda raporlar. Ayrica final
dropna()'yi "kategorik alanlar Unknown/Belirtilmemiş'e cevrilirse" senaryosuyla
SIMULE ederek kac satir/grup geri kazanilabilecegini hesaplar - preprocess.py'ye
DOKUNMADAN (ayri, paralel bir simulasyon fonksiyonu).

Calistirma (ai-model/ calisma dizini olarak): python analyze_dropna_loss.py
"""
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import CURRENT_YEAR, DROP_COLS, TRAIN_PATH, UNKNOWN_FLAG_COLS

# final dropna()'nin baktigi kolonlar (load_clean_train_dataset()'teki SIRAYLA
# ayni: DROP_COLS cikarilmis, degisen/boyali/agir_hasarli ZATEN fillna edilmis,
# yas/km_yil turetilmis HALDEKI dataframe'in TUM kolonlari).
FINAL_DROPNA_WATCH_COLS = [
    'marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
    'yil', 'kilometre', 'yakit_turu', 'vites', 'fiyat',
]

# kullanicinin istedigi 3 kategorili siniflandirma - GERCEK kayip sayilarina
# gore asagida (main() ciktisinda) dogrulanir/tartisilir, burada sadece
# script'in kendi one-siniflandirmasi (rapora yazilir).
FIELD_CLASSIFICATION = {
    'fiyat': 'ZORUNLU (target) - etiketsiz satir egitilemez, drop DOGRU',
    'yil': 'ZORUNLU (numeric/kritik) - yas/km_yil/hierarchical_price yas-egrisi buna bagli',
    'kilometre': 'ZORUNLU (numeric/kritik) - km_yil buna bagli, temel fiyat belirleyici',
    'marka': 'ZORUNLU (join/kategori anahtari) - pratikte NaN orani ~0, gercek kayip kaynagi degil',
    'model': 'ZORUNLU (join/kategori anahtari) - pratikte NaN orani ~0, gercek kayip kaynagi degil',
    'paket': 'KURTARILABILIR (categorical) - "Belirtilmemiş" flag+deger onerilir (degisen_sayisi_bilinmiyor ile AYNI desen)',
    'kasa_turu': 'KURTARILABILIR (categorical) - "Belirtilmemiş" flag+deger onerilir',
    'renk': 'KURTARILABILIR (categorical) - "Belirtilmemiş" flag+deger onerilir',
    'yakit_turu': 'KURTARILABILIR (categorical) - "Belirtilmemiş" flag+deger onerilir (dikkat: dusuk NaN orani, az etkili)',
    'vites': 'KURTARILABILIR (categorical) - "Belirtilmemiş" flag+deger onerilir (dikkat: dusuk NaN orani, az etkili)',
    'motor_hacmi': 'TARTISMALI (numeric ama native-missing destekli) - LightGBM NaN\'i native isliyor (bkz. train.py to_category yorumu), drop yerine NaN birakilabilir',
    'motor_gucu': 'TARTISMALI (numeric ama native-missing destekli) - ayni motor_hacmi mantigi',
}


def trace_pipeline(raw_df):
    """load_clean_train_dataset() ile AYNI sirayla (fiyat q99 -> km -> final
    dropna) calisir ama HER satirin HANGI asamada/HANGI kolon(lar) yuzunden
    dustugunu 'drop_stage'/'drop_cols' olarak isaretler - dusurmez, ISARETLER."""
    df = raw_df.copy()
    df['drop_stage'] = None
    df['drop_cols'] = None

    valid_mm = df['marka'].notna() & df['model'].notna()
    df.loc[~valid_mm, 'drop_stage'] = 'missing_marka_or_model'

    brand_q99 = df.groupby('marka')['fiyat'].transform(lambda s: s.quantile(0.99))
    fails_price = valid_mm & (df['drop_stage'].isna()) & ~(df['fiyat'] <= brand_q99)
    df.loc[fails_price, 'drop_stage'] = 'price_brand_q99'

    still_alive = df['drop_stage'].isna()
    fails_km = still_alive & ~(df['kilometre'] <= 1_000_000)
    df.loc[fails_km, 'drop_stage'] = 'km_over_1m'

    still_alive = df['drop_stage'].isna()
    # degisen/boyali/agir_hasarli fillna - final dropna'da NaN kaynagi olmaktan cikar
    watch = df.loc[still_alive, FINAL_DROPNA_WATCH_COLS]
    na_mask = watch.isna()
    any_na = na_mask.any(axis=1)
    dropna_idx = watch.index[any_na]
    df.loc[dropna_idx, 'drop_stage'] = 'final_dropna'
    df.loc[dropna_idx, 'drop_cols'] = [
        ','.join(watch.columns[na_mask.loc[i]].tolist()) for i in dropna_idx
    ]

    df.loc[still_alive & ~any_na, 'drop_stage'] = 'kept'
    return df


def simulate_recovery(traced_df, recoverable_cols, native_missing_cols=()):
    """'KURTARILABILIR' isaretli kategorik kolonlari 'Belirtilmemiş' ile
    doldurup, 'TARTISMALI' (native_missing_cols) numerik kolonlari dropna
    kontrolunden TAMAMEN cikarip (LightGBM'e NaN olarak birakip) final
    dropna'yi TEKRAR uygulayan simulasyon - preprocess.py DEGISMEZ, sadece
    bu fonksiyon icinde paralel bir kopya uzerinde denenir."""
    sim = traced_df.copy()
    pre_dropna_mask = sim['drop_stage'].isin(['final_dropna', 'kept'])
    if recoverable_cols:
        sim.loc[pre_dropna_mask, recoverable_cols] = sim.loc[pre_dropna_mask, recoverable_cols].fillna('Belirtilmemiş')

    remaining_watch = [c for c in FINAL_DROPNA_WATCH_COLS if c not in recoverable_cols and c not in native_missing_cols]
    watch = sim.loc[pre_dropna_mask, remaining_watch]
    still_na = watch.isna().any(axis=1)
    sim.loc[watch.index, 'sim_kept'] = ~still_na.values
    return sim


def main():
    print('Ham veri okunuyor:', TRAIN_PATH)
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f'  {len(raw)} ham satir')

    traced = trace_pipeline(raw)

    print('\n=== ASAMA BAZLI KAYIP (tum dataset, 308k satir) ===')
    print(traced['drop_stage'].value_counts().to_string())

    # --- 256 hedef grup: onceki rapordan yeniden turetiliyor (raw>0, train=0) ---
    valid = raw.dropna(subset=['marka', 'model']).copy()
    valid['marka'] = valid['marka'].astype(str).str.strip()
    valid['model'] = valid['model'].astype(str).str.strip()
    raw_counts = valid.groupby(['marka', 'model'], observed=True).size()

    kept_groups = set(
        traced[traced['drop_stage'] == 'kept'].dropna(subset=['marka', 'model'])
        .assign(marka=lambda d: d['marka'].astype(str).str.strip(), model=lambda d: d['model'].astype(str).str.strip())
        .groupby(['marka', 'model']).size().index
    )
    target_groups = set(raw_counts.index) - kept_groups
    print(f'\nraw_real_count>0 & train_real_count=0 grup sayisi (dogrulama): {len(target_groups)}')

    target_mask = traced.apply(
        lambda r: (str(r['marka']).strip(), str(r['model']).strip()) in target_groups
        if pd.notna(r['marka']) and pd.notna(r['model']) else False, axis=1,
    )
    target_rows = traced[target_mask]
    print(f'bu gruplara ait TOPLAM ham satir: {len(target_rows)}')
    print('\n256 grubun HAM satirlarinin asama bazli dagilimi:')
    print(target_rows['drop_stage'].value_counts().to_string())

    dropna_rows = target_rows[target_rows['drop_stage'] == 'final_dropna']
    print(f'\nfinal_dropna asamasinda dusen satir sayisi (256 grup icinde): {len(dropna_rows)}')

    print('\n=== KOLON BAZLI DROP NEDENI (final_dropna satirlari icinde, 256 grup) ===')
    col_counts = {c: 0 for c in FINAL_DROPNA_WATCH_COLS}
    col_group_sets = {c: set() for c in FINAL_DROPNA_WATCH_COLS}
    for _, r in dropna_rows.iterrows():
        cols = r['drop_cols'].split(',')
        key = (str(r['marka']).strip(), str(r['model']).strip())
        for c in cols:
            col_counts[c] += 1
            col_group_sets[c].add(key)

    print(f"{'kolon':<15}{'satir_sayisi':<14}{'etkilenen_grup':<16}siniflandirma")
    for c in FINAL_DROPNA_WATCH_COLS:
        print(f"{c:<15}{col_counts[c]:<14}{len(col_group_sets[c]):<16}{FIELD_CLASSIFICATION[c]}")

    categorical_cols = ['paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites']
    numeric_native = ['motor_hacmi', 'motor_gucu']

    scenarios = [
        ('A: sadece kategorik -> Belirtilmemiş', categorical_cols, ()),
        ('B: A + motor_hacmi/motor_gucu native-missing', categorical_cols, numeric_native),
    ]
    for label, cat_cols, num_cols in scenarios:
        print(f'\n=== SIMULASYON {label} (preprocess.py DEGISMEDEN) ===')
        sim = simulate_recovery(traced, cat_cols, num_cols)
        sim_target = sim[target_mask]
        recovered = sim_target[(sim_target['drop_stage'] == 'final_dropna') & (sim_target['sim_kept'] == True)]
        print(f'256 grup icinde final_dropna\'dan kurtarilan satir sayisi: {len(recovered)} / {len(dropna_rows)}')
        recovered_groups = set(zip(recovered['marka'].astype(str).str.strip(), recovered['model'].astype(str).str.strip()))
        print(f'256 gruptan kac tanesi EN AZ 1 gercek satir kazaniyor (train_real_count=0 -> >0): {len(recovered_groups)}')

        all_dropna = sim[sim['drop_stage'] == 'final_dropna']
        all_recovered = all_dropna[all_dropna['sim_kept'] == True]
        print(f'TUM dataset genelinde final_dropna satirlarindan kurtarilan: {len(all_recovered)} / {len(all_dropna)}')
        print(f'TUM dataset genelinde YENI toplam egitim satiri (287402 + kurtarilan): {287402 + len(all_recovered)}')

    out_path = 'reports/dropna_column_loss.csv'
    rows = []
    for c in FINAL_DROPNA_WATCH_COLS:
        rows.append({
            'kolon': c, 'kaybolan_satir': col_counts[c], 'etkilenen_grup': len(col_group_sets[c]),
            'siniflandirma': FIELD_CLASSIFICATION[c],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\nRapor yazildi: {out_path}')


if __name__ == '__main__':
    main()
