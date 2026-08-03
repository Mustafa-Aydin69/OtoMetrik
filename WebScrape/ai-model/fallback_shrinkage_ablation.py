"""Faz 22: fallback_shrinkage.py'deki A/B/C varyantlarinin ablation runner'i.
Faz 21'in (time_holdout_evaluation.py) tespit ettigi zayifligi hedefler: production
(A_current: marka+model -> marka -> global) yeni marka-model kombinasyonlarinda hem
dis holdout'ta hem zaman holdout'ta baseline'a gore TUTARLI kotu tahmin ediyordu.

Bu script SADECE OLCER - production'a (models/lightgbm_final.joblib, hierarchical_price.py)
HICBIR SEY YAZMAZ. Kullanicinin karari acik: "doğrudan production'a eklemeyin, ayrı
ablation yapın" - sonuclar sadece rapor edilir, bir sonraki (ayri, kullanicinin onayiyla
gelecek) adimda production'a alinip alinmayacagina karar verilecek.

Iki DUNYA'da, 4 varyanttan (A_current, B_model_tier, C_shrink_k5, C_shrink_k20) her
biri icin AYRI bir LightGBM egitilir (feature SADECE bu tek kolonda degisir, gerisi
BASELINE_PARAMS/CATEGORICAL_COLS ile production'la BIREBIR AYNI):
  - DIS HOLDOUT: train.py'nin prepare_full_training_data()/prepare_external_holdout()
    pipeline'i (production'in kendi rastgele-holdout degerlendirme yolu, cars1_normalized.csv).
  - ZAMAN HOLDOUT: time_holdout_evaluation.py'nin AYNI 70/15/15 kronolojik split'i
    (bkz. o modulun docstring'i - tarih guvenilirligi/duplicate temizligi ORADA yapildi,
    burada TEKRAR uretilmez, ayni fonksiyonlar import edilir).

Her ikisinde de odak: GENEL, nadir_model(freq<=20), yeni_marka_model_kombinasyonu
segmentleri (Faz 21'in zayif buldugu tam da bu son segment).

Calistirma (ai-model/ calisma dizini olarak): python fallback_shrinkage_ablation.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from fallback_shrinkage import VARIANTS, attach_variant_feature, build_variant_lookup, compute_oof_variant, lookup_variant
from hierarchical_price import FEATURE_COLUMN
from time_holdout_evaluation import (
    FEATURE_COLS, RARE_MODEL_FREQ_THRESHOLD, load_clean_with_dates, make_time_split,
    regression_stats, remove_cross_split_duplicates, to_category_fit,
)
from train import BASELINE_PARAMS, prepare_external_holdout, prepare_full_training_data

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'fallback_shrinkage_ablation_report.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'fallback_shrinkage_ablation_summary.txt')


def fmt_stats(label, s):
    return (f'{label:42s} n={s["n"]:>7,} MAE={s["mae"]:>9,.0f} RMSE={s["rmse"]:>9,.0f} '
            f'R2={s["r2"]:>7.4f} MedAE={s["medae"]:>9,.0f} bias={s["bias"]:>+9,.0f} sMAPE=%{s["smape"]:>5.1f}')


def build_segments(X, model_freq_train, train_models_seen, train_combos_seen):
    segs = {'GENEL': np.ones(len(X), dtype=bool)}
    segs[f'nadir_model(freq<={RARE_MODEL_FREQ_THRESHOLD})'] = (
        X['model'].map(lambda m: model_freq_train.get(m, 0)).values <= RARE_MODEL_FREQ_THRESHOLD)
    combos = list(zip(X['marka'], X['model']))
    segs['yeni_marka_model_kombinasyonu'] = np.array([c not in train_combos_seen for c in combos])
    return segs


def tier_breakdown(X, lookup):
    """Her satirin hangi fallback katmanindan (brand_model/model/brand/global/
    brand_model_shrunk) deger aldiginin dagilimi - "neden iyilesti/kotulesti" sorusuna
    dogrudan cevap verir (orn. B varyantinin 'model' katmani ne kadar kullaniliyor)."""
    sources = [lookup_variant(m, mo, lookup)[1] for m, mo in zip(X['marka'], X['model'])]
    return pd.Series(sources).value_counts(normalize=True).mul(100).round(1).to_dict()


def run_world(world_name, X_train, y_train, eval_sets, csv_rows, lines):
    """eval_sets: {split_label: (X, y)}. X_train/y_train FEATURE_COLUMN (production'in
    kendi hierarchical feature'i) icermemeli - her varyant kendi kolonunu ekler."""
    model_freq_train = X_train['model'].value_counts().to_dict()
    train_models_seen = set(X_train['model'].astype(object))
    train_combos_seen = set(zip(X_train['marka'], X_train['model']))

    for variant in VARIANTS:
        lines.append(f'--- {world_name} / {variant} ---')
        oof_values = compute_oof_variant(X_train, y_train, variant)
        X_train_v = X_train.assign(**{'brand_model_median_price_variant': oof_values})
        lookup = build_variant_lookup(X_train, y_train, variant)

        model = LGBMRegressor(**BASELINE_PARAMS)
        model.fit(X_train_v, y_train)

        for split_label, (X_eval, y_eval) in eval_sets.items():
            X_eval_v = attach_variant_feature(X_eval, lookup)
            preds = model.predict(X_eval_v)
            segs = build_segments(X_eval, model_freq_train, train_models_seen, train_combos_seen)
            for seg_name, mask in segs.items():
                if mask.sum() == 0:
                    continue
                stats = regression_stats(y_eval.values[mask], preds[mask])
                lines.append(fmt_stats(f'{split_label}/{seg_name}', stats))
                csv_rows.append({'world': world_name, 'variant': variant, 'split': split_label,
                                 'segment': seg_name, **stats})
            tiers = tier_breakdown(X_eval, lookup)
            lines.append(f'  {split_label} katman kullanim dagilimi (%): {tiers}')
        lines.append('')


def main():
    lines = []
    csv_rows = []

    # === DIS HOLDOUT dunyasi ===
    lines.append('=== DIS HOLDOUT (cars1_normalized.csv) - production pipeline, sadece hier. ozellik varyanti degisiyor ===')
    X_full, y_full = prepare_full_training_data()
    X_full = X_full.drop(columns=[FEATURE_COLUMN])  # production'in KENDI hier. ozelligini disla - her varyant kendi ekleyecek
    X_holdout, y_holdout = prepare_external_holdout(X_full, y_full)  # FEATURE_COLUMN X_full de yok -> otomatik eklenmez
    run_world('dis_holdout', X_full, y_full, {'holdout': (X_holdout, y_holdout)}, csv_rows, lines)

    # === ZAMAN HOLDOUT dunyasi (bkz. time_holdout_evaluation.py - AYNI split fonksiyonlari) ===
    lines.append('=== ZAMAN HOLDOUT (arabam kronolojik 70/15/15 split, bkz. time_holdout_evaluation.py) ===')
    df = load_clean_with_dates()
    df, _, _ = make_time_split(df)
    df, n_leak = remove_cross_split_duplicates(df)
    lines.append(f'(zaman split sizinti temizligi: {n_leak} satir val/test ten cikarildi - bkz. time_holdout_evaluation.py Madde 2)\n')

    train_df = df[df['time_split'] == 'train']
    val_df = df[df['time_split'] == 'val']
    test_df = df[df['time_split'] == 'test']
    X_train_t = train_df[FEATURE_COLS].copy()
    y_train_t = train_df['fiyat'].copy()
    X_val_t = val_df[FEATURE_COLS].copy()
    y_val_t = val_df['fiyat'].copy()
    X_test_t = test_df[FEATURE_COLS].copy()
    y_test_t = test_df['fiyat'].copy()
    X_train_t, X_val_t, X_test_t = to_category_fit(X_train_t, X_val_t, X_test_t)

    run_world('zaman_holdout', X_train_t, y_train_t,
              {'val': (X_val_t, y_val_t), 'test': (X_test_t, y_test_t)}, csv_rows, lines)

    # === Karsilastirma tablosu: A'ya gore % degisim (GENEL + yeni kombinasyon) ===
    report_df = pd.DataFrame(csv_rows)
    lines.append('=== A_current e gore % MAE degisimi (negatif = iyilesme) ===')
    for world in report_df['world'].unique():
        for split in report_df.loc[report_df['world'] == world, 'split'].unique():
            for seg in ('GENEL', f'nadir_model(freq<={RARE_MODEL_FREQ_THRESHOLD})', 'yeni_marka_model_kombinasyonu'):
                sub = report_df[(report_df['world'] == world) & (report_df['split'] == split) & (report_df['segment'] == seg)]
                if sub.empty or 'A_current' not in sub['variant'].values:
                    continue
                base_mae = sub.loc[sub['variant'] == 'A_current', 'mae'].iloc[0]
                row_str = f'{world}/{split}/{seg}: A_current MAE={base_mae:,.0f}  '
                for variant in VARIANTS[1:]:
                    v_row = sub[sub['variant'] == variant]
                    if v_row.empty:
                        continue
                    v_mae = v_row['mae'].iloc[0]
                    pct = 100 * (v_mae - base_mae) / base_mae
                    row_str += f'{variant}={v_mae:,.0f}(%{pct:+.1f})  '
                lines.append(row_str)
    lines.append('')

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    report_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    print('\n'.join(lines))
    print(f'CSV: {CSV_PATH}')
    print(f'Ozet: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
