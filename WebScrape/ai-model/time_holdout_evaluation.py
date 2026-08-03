"""Faz 21: brand_model_median_price (Faz 20) icin ZAMAN BAZLI genelleme testi.
Rastgele holdout'ta (Faz 19/20) dogrulanan kazancin, farkli takvim donemlerine
de genellendigini/genellenmedigini olcer. Bu script SADECE DEGERLENDIRME
yapar - models/lightgbm_final.joblib'e (production artefakti) HIC DOKUNMAZ.

=== 1) Tarih alani guvenilirligi (bkz. print_date_field_reliability()) ===
train_dataset.csv'de TEK gercek tarih alani 'scraped_at' - ama kaynaga gore
COK FARKLI anlam/guvenilirlik tasiyor:
  - arabam-*  (%82.5, n=237.229): kendi CANLI kazimamizin scrape ZAMANI
    (ilan yayinlanma tarihi DEGIL) - yogun, surekli, 2026-07-07..2026-08-03
    arasi ~27 gunluk GERCEK bir zaman ekseni. GUVENILIR (recency proxy).
  - kaggle-ar (%0.8, n=2.361): arabalar.csv'nin KENDI 'Ilan Tarihi' alani
    (prepare_train_dataset.py load_arabalar() - gercek ilan tarihi), ama
    ~2025-06 donemine ait - arabam penceresinden 13+ ay ONCE, TAMAMEN
    AYRIK/eski bir donem. GUVENILIR ama arabam ile AYNI zaman ekseninde
    DEGIL (farkli "checkpoint").
  - kaggle-ab (%16.7, n=48.005): scraped_at HIC YOK (prepare_train_dataset.py
    load_araba_bilgileri() bilerek None birakiyor - "sahte tarih uydurmak
    yerine bilinmiyor"). GUVENILIR DEGIL - bu satirlarin gercek zamani
    OLCULEMEZ, split'e dahil edilemez.

KARAR: zaman split SADECE arabam'in gercek scraped_at ekseninde kurulur.
kaggle-ar (kanitlanabilir sekilde arabam'dan cok daha eski) ve kaggle-ab
(tarihsiz) satirlari HER ZAMAN train'e alinir - hicbiri val/test'e
GIREMEZ (dogru zaman sirasi iddia edilemeyen bir satiri "yeni" olarak
degerlendirmek sahte-genelleme testi verir).

=== 2) Split ===
arabam scraped_at ekseninde ilk %70 -> train, sonraki %15 -> val, son %15
-> test (+ kaggle-ab/kaggle-ar HER ZAMAN train). Ayni ilan_id'nin (yeniden
kazinmis kayit) birden fazla split'e dusup dusmedigi VE marka+model+yil+
km+fiyat TAM esleyen "ayni araba" kayitlarinin train ile val/test arasinda
sizip sizmedigi ayrica kontrol edilir/temizlenir (bkz. remove_cross_split_
duplicates()).

=== 3) Iki model ===
baseline: brand_model_median_price YOK. production: VAR - train icin 5-fold
OOF (hierarchical_price.py ile AYNI fonksiyon), val/test icin SADECE train
donemi verisinden kurulan fold'suz lookup (build_price_lookup + attach_
lookup_feature) - val/test fiyatlari HICBIR hesaba girmez.

Calistirma (ai-model/ calisma dizini olarak): python time_holdout_evaluation.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hierarchical_price import (
    FEATURE_COLUMN, OOF_N_SPLITS, OOF_SEED, attach_lookup_feature, attach_oof_feature,
    build_price_lookup,
)
from preprocess import CURRENT_YEAR, TRAIN_PATH, UNKNOWN_FLAG_COLS
from train import BASELINE_PARAMS, CATEGORICAL_COLS

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
REPORT_DIR = os.path.join(BASE_DIR, 'data', 'output', 'time_holdout')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'time_holdout_report.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'time_holdout_summary.txt')

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # kalan 0.15 -> test
RARE_MODEL_FREQ_THRESHOLD = 20
DUP_KEY_COLS = ['marka', 'model', 'yil', 'kilometre', 'fiyat']
FEATURE_COLS = [
    'marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu', 'yil',
    'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli',
    'degisen_sayisi_bilinmiyor', 'boyali_sayisi_bilinmiyor', 'yas', 'km_yil',
]
# Freshness pencereleri gorev talebindeki 3/6/12 ay yerine veri yogunlugunun izin
# verdigi esdeger araliklara indirgendi - arabam'in TUM gercek kazima gecmisi
# sadece ~27 gun (bkz. print_date_field_reliability()); aylik pencereler bu
# veriyle HENUZ olculemez, bu acikca raporlanir (bkz. run_freshness_analysis()).
FRESHNESS_WINDOWS = {'son_1_hafta': pd.Timedelta(days=7), 'son_2_hafta': pd.Timedelta(days=14)}


def assign_source(ilan_id: str) -> str:
    if ilan_id.startswith('kaggle-ab-'):
        return 'kaggle-ab'
    if ilan_id.startswith('kaggle-ar-'):
        return 'kaggle-ar'
    return 'arabam'


# preprocess.load_clean_train_dataset() ile BIREBIR AYNI temizleme adimlari - TEK
# fark: scraped_at/ilan_id split icin KORUNUR (drop edilmez) ve son dropna() bu iki
# alanin NaN'ligini satir cikarma SEBEBI SAYMAZ (DROP_COLS'daki mantikla ayni:
# scraped_at zaten kaynaga-ozgu bir alan, MODEL OZELLIGI degil). Bu, production'daki
# 287.595 satirlik ayni temiz kumeyi (+ split icin gereken 2 metadata kolonu) uretir.
def load_clean_with_dates():
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    df = df[df['fiyat'] <= df['fiyat'].quantile(0.99)]
    df = df[df['kilometre'] <= 1_000_000]

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)
    df['agir_hasarli'] = df['agir_hasarli'].fillna(0)
    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)

    model_cols = [c for c in df.columns if c not in ('scraped_at', 'arac_turu')]
    before = len(df)
    df = df.dropna(subset=model_cols).reset_index(drop=True)
    print(f'[load_clean_with_dates] preprocess.py ile ayni filtre: {before - len(df)}/{before} '
          f'satir cikarildi (scraped_at eksikligi SEBEP SAYILMADI)')

    df = df.drop(columns=['arac_turu'])
    df['source'] = df['ilan_id'].astype(str).map(assign_source)
    df['scraped_at_parsed'] = pd.to_datetime(df['scraped_at'], utc=True, errors='coerce')
    return df


def print_date_field_reliability(df):
    lines = ['=== 1) Tarih alani guvenilirligi ===']
    for src in ('arabam', 'kaggle-ar', 'kaggle-ab'):
        sub = df[df['source'] == src]
        n_dated = sub['scraped_at_parsed'].notna().sum()
        if n_dated > 0:
            lines.append(f'{src:10s} n={len(sub):>7,}  tarihli={n_dated:>7,} '
                         f'({sub["scraped_at_parsed"].min()} -> {sub["scraped_at_parsed"].max()})')
        else:
            lines.append(f'{src:10s} n={len(sub):>7,}  tarihli=0 (scraped_at HIC YOK - GUVENILIR DEGIL, split disi)')
    lines.append('KARAR: split SADECE arabam scraped_at ekseninde kurulur (gercek, yogun, surekli zaman '
                 'ekseni). kaggle-ar tarihli ama arabam penceresinden 13+ ay ONCEsine ait (AYRIK checkpoint) '
                 '- train-only. kaggle-ab tarihsiz - train-only (bkz. modul docstring Madde 1).')
    lines.append('')
    return lines


def make_time_split(df):
    arabam = df[df['source'] == 'arabam']
    q_train, q_val = arabam['scraped_at_parsed'].quantile([TRAIN_FRAC, TRAIN_FRAC + VAL_FRAC])

    df = df.copy()
    df['time_split'] = 'train'  # varsayilan: kaggle-ab/kaggle-ar (ve tanim geregi tum non-arabam) train
    arabam_split = np.select(
        [arabam['scraped_at_parsed'] <= q_train, arabam['scraped_at_parsed'] <= q_val],
        ['train', 'val'], default='test')
    df.loc[arabam.index, 'time_split'] = arabam_split
    return df, q_train, q_val


def check_duplicate_ilan_id_across_splits(df):
    arabam = df[df['source'] == 'arabam']
    dup = arabam[arabam.duplicated('ilan_id', keep=False)]
    if len(dup) == 0:
        return 0, 0
    cross = dup.groupby('ilan_id')['time_split'].nunique()
    return (cross > 1).sum(), len(cross)


# Gorev talebi: "ayni aracin ... donemler arasinda tasinmasini kontrol edin". ilan_id
# tekrarlari (yeniden kazima) AYRI kontrol edilir (bkz. check_duplicate_ilan_id_across_splits) -
# bu fonksiyon FARKLI kaynak/ilan_id'li ama marka+model+yil+km+fiyat TAM esleyen (muhtemelen
# ayni fiziksel araç/ilanin baska bir kaynaktaki/zamandaki kopyasi) satirlari bulur; train'de
# GORULMUS bir kombinasyonun val/test'te AYNEN tekrarlanmasi, o satirin gercek fiyatinin zaten
# train'de mevcut olmasi anlamina gelir (sizinti) - bu yuzden val/test'ten CIKARILIR (train'de
# kalir, sadece degerlendirmeden dislanir).
def remove_cross_split_duplicates(df):
    df = df.copy()
    df['_dupkey'] = df[DUP_KEY_COLS].astype(str).agg('|'.join, axis=1)
    train_keys = set(df.loc[df['time_split'] == 'train', '_dupkey'])
    leak_mask = (df['time_split'] != 'train') & (df['_dupkey'].isin(train_keys))
    n_leak = int(leak_mask.sum())
    df = df[~leak_mask].drop(columns=['_dupkey']).reset_index(drop=True)
    return df, n_leak


def to_category_fit(X_train, *others):
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


def regression_stats(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    denom = (np.abs(y_true) + np.abs(y_pred))
    smape = float(np.mean(np.where(denom > 0, 2 * abs_err / denom, 0.0)) * 100)
    return {
        'n': len(y_true), 'mae': float(abs_err.mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
        'r2': float(r2), 'medae': float(np.median(abs_err)), 'bias': float(err.mean()), 'smape': smape,
    }


def fmt_stats(label, s):
    return (f'{label:36s} n={s["n"]:>7,} MAE={s["mae"]:>9,.0f} RMSE={s["rmse"]:>9,.0f} '
            f'R2={s["r2"]:>7.4f} MedAE={s["medae"]:>9,.0f} bias={s["bias"]:>+9,.0f} sMAPE=%{s["smape"]:>5.1f}')


def build_segments(X, y, model_freq_train, train_models_seen, train_combos_seen, price_bins, price_labels):
    segs = {'GENEL': np.ones(len(y), dtype=bool)}
    segs[f'nadir_model(freq<={RARE_MODEL_FREQ_THRESHOLD})'] = (
        X['model'].map(lambda m: model_freq_train.get(m, 0)).values <= RARE_MODEL_FREQ_THRESHOLD)
    segs['300+HP'] = X['motor_gucu'].values >= 300
    combos = list(zip(X['marka'], X['model']))
    segs['yeni_marka_model_kombinasyonu'] = np.array([c not in train_combos_seen for c in combos])
    segs['ilk_kez_gorulen_model'] = ~X['model'].astype(object).isin(train_models_seen).values
    price_bin_idx = pd.cut(y, bins=price_bins, labels=price_labels)
    for label in price_labels:
        segs[f'fiyat_{label}'] = (price_bin_idx == label).values
    if (X['agir_hasarli'] == 1).any():
        segs['agir_hasarli=Evet'] = X['agir_hasarli'].values == 1
    return segs


def main():
    df = load_clean_with_dates()
    df, q_train, q_val = make_time_split(df)

    reliability_lines = print_date_field_reliability(df)

    n_dup_id_crossing, n_dup_id_groups = check_duplicate_ilan_id_across_splits(df)
    df, n_leak = remove_cross_split_duplicates(df)

    split_lines = ['=== 2) Split ===']
    split_lines.append(f'train kesim tarihi (q{TRAIN_FRAC:.0%}): {q_train}')
    split_lines.append(f'val kesim tarihi   (q{TRAIN_FRAC + VAL_FRAC:.0%}): {q_val}')
    split_lines.append(f'ayni ilan_id birden fazla split e dusen grup sayisi: {n_dup_id_crossing} / {n_dup_id_groups}')
    split_lines.append(f'marka+model+yil+km+fiyat TAM esleyen, train de zaten GORULMUS oldugu icin '
                       f'val/test ten CIKARILAN satir sayisi (sizinti temizligi): {n_leak}')
    for split in ('train', 'val', 'test'):
        sub = df[df['time_split'] == split]
        src_counts = sub['source'].value_counts().to_dict()
        split_lines.append(f'{split:6s} n={len(sub):>7,}  kaynak dagilimi={src_counts}')
    split_lines.append('')
    print('\n'.join(reliability_lines + split_lines))

    train_df = df[df['time_split'] == 'train']
    val_df = df[df['time_split'] == 'val']
    test_df = df[df['time_split'] == 'test']

    X_train, y_train = train_df[FEATURE_COLS].copy(), train_df['fiyat'].copy()
    X_val, y_val = val_df[FEATURE_COLS].copy(), val_df['fiyat'].copy()
    X_test, y_test = test_df[FEATURE_COLS].copy(), test_df['fiyat'].copy()
    X_train, X_val, X_test = to_category_fit(X_train, X_val, X_test)

    print('=== 3) Model egitimi (zaman-train uzerinde) ===')
    baseline_model = LGBMRegressor(**BASELINE_PARAMS)
    baseline_model.fit(X_train, y_train)

    X_train_hier, _ = attach_oof_feature(X_train, y_train)
    lookup = build_price_lookup(X_train, y_train)  # SADECE train donemi - val/test hicbir hesaba girmez
    X_val_hier = attach_lookup_feature(X_val, lookup)
    X_test_hier = attach_lookup_feature(X_test, lookup)

    production_model = LGBMRegressor(**BASELINE_PARAMS)
    production_model.fit(X_train_hier, y_train)
    print('baseline ve production modelleri egitildi.\n')

    preds = {
        ('baseline', 'val'): baseline_model.predict(X_val),
        ('baseline', 'test'): baseline_model.predict(X_test),
        ('production', 'val'): production_model.predict(X_val_hier),
        ('production', 'test'): production_model.predict(X_test_hier),
    }

    model_freq_train = X_train['model'].value_counts().to_dict()
    train_models_seen = set(X_train['model'].astype(object))
    train_combos_seen = set(zip(X_train['marka'], X_train['model']))
    price_bins = list(y_train.quantile([0, .25, .5, .75, 1.0]).values)
    price_bins[0] -= 1
    price_bins[-1] += 1
    price_labels = ['Q1_en_ucuz', 'Q2', 'Q3', 'Q4_en_pahali']

    csv_rows = []
    metric_lines = ['=== 4) Metrikler (val/test x baseline/production x segment) ===']
    for split_name, X_split, y_split in (('val', X_val, y_val), ('test', X_test, y_test)):
        segs = build_segments(X_split, y_split, model_freq_train, train_models_seen,
                               train_combos_seen, price_bins, price_labels)
        metric_lines.append(f'--- {split_name} ---')
        for model_name in ('baseline', 'production'):
            p = preds[(model_name, split_name)]
            for seg_name, mask in segs.items():
                if mask.sum() == 0:
                    continue
                stats = regression_stats(y_split.values[mask], p[mask])
                metric_lines.append(fmt_stats(f'{model_name}/{seg_name}', stats))
                csv_rows.append({'split': split_name, 'model': model_name, 'segment': seg_name, **stats})
        metric_lines.append('')

    # === 5) Piyasa kaymasi ===
    drift_lines = ['=== 5) Piyasa kaymasi analizi ===']
    train_arabam = train_df[train_df['source'] == 'arabam']
    drift_lines.append(f'train(arabam,tarihli) fiyat: n={len(train_arabam):,} '
                       f'ort={train_arabam["fiyat"].mean():,.0f} medyan={train_arabam["fiyat"].median():,.0f}')
    drift_lines.append(f'val fiyat:                   n={len(val_df):,} '
                       f'ort={val_df["fiyat"].mean():,.0f} medyan={val_df["fiyat"].median():,.0f}')
    drift_lines.append(f'test fiyat:                  n={len(test_df):,} '
                       f'ort={test_df["fiyat"].mean():,.0f} medyan={test_df["fiyat"].median():,.0f}')

    train_bm_median = train_arabam.groupby(['marka', 'model'], observed=True)['fiyat'].median()
    train_bm_count = train_arabam.groupby(['marka', 'model'], observed=True)['fiyat'].size()
    test_bm_median = test_df.groupby(['marka', 'model'], observed=True)['fiyat'].median()
    test_bm_count = test_df.groupby(['marka', 'model'], observed=True)['fiyat'].size()
    common = [k for k in train_bm_median.index.intersection(test_bm_median.index)
              if train_bm_count.get(k, 0) >= 5 and test_bm_count.get(k, 0) >= 5]
    if common:
        pct_change = pd.Series({k: 100 * (test_bm_median[k] - train_bm_median[k]) / train_bm_median[k]
                                 for k in common})
        drift_lines.append(f'\nmarka+model medyan fiyati train->test karsilastirmasi (n>=5 iki tarafta da, '
                           f'{len(common)} kombinasyon):')
        drift_lines.append(f'  ortalama degisim: %{pct_change.mean():+.1f}  medyan degisim: %{pct_change.median():+.1f}')
        drift_lines.append(f'  en cok DUSEN 5: {pct_change.nsmallest(5).round(1).to_dict()}')
        drift_lines.append(f'  en cok YUKSELEN 5: {pct_change.nlargest(5).round(1).to_dict()}')
    else:
        pct_change = pd.Series(dtype=float)
        drift_lines.append('yeterli ortak (n>=5) marka+model kombinasyonu bulunamadi (kucuk pencere).')

    bias_base_val = regression_stats(y_val.values, preds[('baseline', 'val')])['bias']
    bias_prod_val = regression_stats(y_val.values, preds[('production', 'val')])['bias']
    bias_base_test = regression_stats(y_test.values, preds[('baseline', 'test')])['bias']
    bias_prod_test = regression_stats(y_test.values, preds[('production', 'test')])['bias']
    drift_lines.append(f'\nsistematik bias (pred-gercek) egilimi:')
    drift_lines.append(f'  baseline:   val={bias_base_val:+,.0f}  test={bias_base_test:+,.0f}')
    drift_lines.append(f'  production: val={bias_prod_val:+,.0f}  test={bias_prod_test:+,.0f}')
    drift_direction = ('production bias val->test negatif yonde BUYUYOR (dusuk-tahmin egilimi guclenmis, '
                       'stale lookup riski ile TUTARLI)' if bias_prod_test < bias_prod_val - 1000 else
                       'production bias val->test belirgin sekilde kotulesmiyor')
    drift_lines.append(f'  yorum: {drift_direction}')
    drift_lines.append('')

    # === 6) Freshness ===
    fresh_lines = ['=== 6) Freshness analizi (lookup penceresi ablation - MODEL SABIT, sadece val/test '
                   'skorlama lookup penceresi degisiyor) ===',
                   'NOT: gorev talebindeki 3/6/12 ay pencereleri bu veri ile OLCULEMEZ - arabam in TUM '
                   'gercek kazima gecmisi sadece ~%d gun (bkz. Madde 1). Esdeger, veri yogunluguna uygun '
                   'pencereler kullanildi: son 1 hafta / son 2 hafta / tum train donemi (~%d gun).' % (
                       (train_arabam['scraped_at_parsed'].max() - train_arabam['scraped_at_parsed'].min()).days,
                       (train_arabam['scraped_at_parsed'].max() - train_arabam['scraped_at_parsed'].min()).days)]
    max_train_date = train_arabam['scraped_at_parsed'].max()
    freshness_results = []
    windows_ordered = [('tum_train_gecmisi', None)] + list(FRESHNESS_WINDOWS.items())
    for label, window in windows_ordered:
        if window is None:
            X_src, y_src = X_train, y_train
        else:
            cutoff = max_train_date - window
            # .values tz-aware datetime64'u tz-naive'e cevirip cutoff (tz-aware Timestamp) ile
            # karsilastirmayi TypeError'a dusuruyordu - pandas Series karsilastirmasi (tz KORUNUR)
            # kullanilir; X_train/train_df AYNI (reset edilmemis) index'i paylastigi icin boolean
            # Series ile indexleme dogru hizalanir.
            keep_mask = (train_df['source'] != 'arabam') | (train_df['scraped_at_parsed'] >= cutoff)
            X_src, y_src = X_train[keep_mask], y_train[keep_mask]
        w_lookup = build_price_lookup(X_src, y_src)
        X_val_w = attach_lookup_feature(X_val, w_lookup)
        X_test_w = attach_lookup_feature(X_test, w_lookup)
        s_val = regression_stats(y_val.values, production_model.predict(X_val_w))
        s_test = regression_stats(y_test.values, production_model.predict(X_test_w))
        freshness_results.append((label, len(X_src), s_val, s_test))
        fresh_lines.append(f'{label:20s} lookup_n={len(X_src):>7,}  ' + fmt_stats('val', s_val).strip() +
                           '  |  ' + fmt_stats('test', s_test).strip())
        csv_rows.append({'split': 'val', 'model': f'production_lookup_{label}', 'segment': 'GENEL', **s_val})
        csv_rows.append({'split': 'test', 'model': f'production_lookup_{label}', 'segment': 'GENEL', **s_test})
    best_test = min(freshness_results, key=lambda r: r[3]['mae'])
    fresh_lines.append(f'\nen dusuk test MAE veren pencere: {best_test[0]} (lookup_n={best_test[1]:,}, '
                       f'test MAE={best_test[3]["mae"]:,.0f})')
    fresh_lines.append('')

    # === 7) Kabul kriteri ===
    accept_lines = ['=== 7) Kabul kriteri degerlendirmesi ===']
    for split_name, y_split in (('val', y_val), ('test', y_test)):
        base_overall = regression_stats(y_split.values, preds[('baseline', split_name)])
        prod_overall = regression_stats(y_split.values, preds[('production', split_name)])
        improved = prod_overall['mae'] <= base_overall['mae']
        accept_lines.append(f'{split_name}: baseline MAE={base_overall["mae"]:,.0f} vs production '
                            f'MAE={prod_overall["mae"]:,.0f} -> {"GECTI (production >= baseline)" if improved else "BASARISIZ (production baseline dan KOTU)"} '
                            f'(%{100 * (base_overall["mae"] - prod_overall["mae"]) / base_overall["mae"]:+.2f})')
        segs = build_segments(X_val if split_name == 'val' else X_test, y_split, model_freq_train,
                              train_models_seen, train_combos_seen, price_bins, price_labels)
        combo_mask = segs['yeni_marka_model_kombinasyonu']
        if combo_mask.sum() > 0:
            base_combo = regression_stats(y_split.values[combo_mask], preds[('baseline', split_name)][combo_mask])
            prod_combo = regression_stats(y_split.values[combo_mask], preds[('production', split_name)][combo_mask])
            collapse = prod_combo['mae'] > base_combo['mae'] * 1.5
            accept_lines.append(f'  yeni marka-model kombinasyonlari (n={combo_mask.sum()}): baseline '
                                f'MAE={base_combo["mae"]:,.0f} production MAE={prod_combo["mae"]:,.0f} -> '
                                f'{"COKUS RISKI (>%50 kotu)" if collapse else "cokus YOK"}')
    accept_lines.append('')

    report_lines = reliability_lines + split_lines + metric_lines + drift_lines + fresh_lines + accept_lines
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    report_df = pd.DataFrame(csv_rows)
    report_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    # --- gorsel: freshness MAE karsilastirmasi ---
    plt.figure(figsize=(7, 4))
    labels_plot = [r[0] for r in freshness_results]
    val_mae = [r[2]['mae'] for r in freshness_results]
    test_mae = [r[3]['mae'] for r in freshness_results]
    x = np.arange(len(labels_plot))
    plt.bar(x - 0.2, val_mae, width=0.4, label='val')
    plt.bar(x + 0.2, test_mae, width=0.4, label='test')
    plt.xticks(x, labels_plot, rotation=20)
    plt.ylabel('MAE (TL)')
    plt.title('Lookup freshness penceresi vs MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'freshness_mae.png'), dpi=120)
    plt.close()

    # --- gorsel: train/val/test fiyat dagilimi (piyasa kaymasi) ---
    plt.figure(figsize=(7, 4))
    plt.boxplot([train_arabam['fiyat'].clip(upper=train_arabam['fiyat'].quantile(0.99)),
                val_df['fiyat'].clip(upper=val_df['fiyat'].quantile(0.99)),
                test_df['fiyat'].clip(upper=test_df['fiyat'].quantile(0.99))],
               tick_labels=['train(arabam)', 'val', 'test'])
    plt.ylabel('fiyat (TL, P99 kirpilmis)')
    plt.title('Zaman split i boyunca fiyat dagilimi (piyasa kaymasi)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'price_drift_boxplot.png'), dpi=120)
    plt.close()

    if len(pct_change) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(pct_change.clip(-80, 150), bins=30)
        plt.xlabel('marka+model medyan fiyat degisimi train->test (%)')
        plt.ylabel('kombinasyon sayisi')
        plt.title('Marka+model medyan fiyat kaymasi dagilimi')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'brand_model_median_drift_hist.png'), dpi=120)
        plt.close()

    print('\n'.join(metric_lines + drift_lines + fresh_lines + accept_lines))
    print(f'CSV: {CSV_PATH}')
    print(f'Ozet: {SUMMARY_PATH}')
    print(f'Gorseller: {REPORT_DIR}')


if __name__ == '__main__':
    main()
