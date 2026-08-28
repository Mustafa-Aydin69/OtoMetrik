"""Faz 34 - Maserati/Porsche (tum modeller) + Mercedes-Benz/Audi/Cadillac
(weak modeller, train_real_count<10) icin brand_model seviyesi sentetik
aday analizi. SADECE ANALIZ - sentetik uretmez, retrain/artifact degisikligi
YAPMAZ.

reports/synthetic_candidates_full.csv (Faz30) HALA GECERLI (hierarchical_price/
hp_support hep real-only) - bu script ONA ek olarak: (1) guncel production
artifact ile per-model PROBE hatasi, (2) engine/body_type/generation tutarlilik
HEURISTIKLERI hesaplar.

Calistirma (ai-model/ calisma dizini olarak): python analyze_third_wave_candidates.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import load_clean_train_dataset
from train import CATEGORICAL_COLS, apply_saved_categories
import hierarchical_price as hp

BASE_DIR = os.path.dirname(__file__)
CANDIDATES_PATH = os.path.join(BASE_DIR, 'reports', 'synthetic_candidates_full.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

PRIMARY_BRANDS = ['Maserati', 'Porsche']
SECONDARY_BRANDS = ['Mercedes - Benz', 'Audi', 'Cadillac']
WEAK_THRESHOLD = 10

ENGINE_CV_HETEROJEN = 0.35
ENGINE_CV_INCELE = 0.15


def engine_consistency(g):
    mh = g['motor_hacmi'].dropna()
    if len(mh) < 2:
        return 'N/A (tek/az veri)'
    cv = mh.std() / mh.mean() if mh.mean() else 0
    if cv > ENGINE_CV_HETEROJEN:
        return f'HETEROJEN (cv={cv:.2f})'
    if cv > ENGINE_CV_INCELE:
        return f'INCELE (cv={cv:.2f})'
    return f'OK (cv={cv:.2f})'


def body_type_consistency(g):
    kasa = g['kasa_turu'].dropna().unique()
    n = len(kasa)
    if n <= 1:
        return f'OK ({list(kasa)})'
    if n == 2:
        return f'COKLU-VARYANT ({list(kasa)})'
    return f'HETEROJEN ({list(kasa)})'


def generation_consistency(price_age_corr, price_cv):
    if price_age_corr is not None and not pd.isna(price_age_corr) and price_age_corr <= -0.3:
        return f'OK (yasla aciklanan, corr={price_age_corr:.2f})'
    if pd.notna(price_cv) and price_cv <= 0.4:
        return f'OK (dusuk varyans, cv={price_cv:.2f})'
    return f'INCELE (corr={price_age_corr}, cv={price_cv})'


def probe_row_and_error(marka, model, clean_df, artifact, hp_lookup):
    g = clean_df[(clean_df['marka'] == marka) & (clean_df['model'] == model)]
    if len(g) == 0:
        return None
    g_ok = g[g['motor_gucu'].notna()]
    g_pick = g_ok if len(g_ok) > 0 else g
    idx = (g_pick['fiyat'] - g_pick['fiyat'].median()).abs().idxmin()
    rep = g_pick.loc[idx]
    row = pd.DataFrame([rep]).drop(columns=['fiyat', 'ilan_id'], errors='ignore')
    row = row.reindex(columns=artifact['feature_columns'])
    for c in CATEGORICAL_COLS:
        row[c] = row[c].astype('category').cat.set_categories(artifact['category_sets'][c])
    yas = row['yas'].iloc[0]
    hp_val, _, _ = hp.lookup_price(marka, model, yas, hp_lookup)
    row[hp.FEATURE_COLUMN] = hp_val
    row_aligned = apply_saved_categories(row, artifact)
    pred = float(artifact['model'].predict(row_aligned)[0])
    actual = float(rep['fiyat'])
    return 100 * abs(pred - actual) / actual


def recommend_count(n, safe):
    if n <= 2 or not safe:
        return 0
    if n > 20:
        return 0
    if 3 <= n <= 5:
        return 6
    if 6 <= n <= 12:
        return 5
    if 13 <= n <= 20:
        return 3
    return 0


def priority(n, safe, probe_err, is_premium):
    if n <= 2 or not safe:
        return 'REJECT'
    if n > 20:
        return 'NO_NEED'
    score = 0
    if 3 <= n <= 12:
        score += 2
    if probe_err is not None and probe_err >= 20:
        score += 2
    elif probe_err is not None and probe_err >= 10:
        score += 1
    if is_premium:
        score += 1
    if score >= 4:
        return 'HIGH'
    if score >= 2:
        return 'MEDIUM'
    return 'LOW'


def build_table(brands, only_weak, cand_df, clean, artifact, hp_lookup, premium_threshold):
    rows = []
    for marka in brands:
        sub = cand_df[cand_df['marka'] == marka]
        if only_weak:
            sub = sub[sub['train_real_count'] < WEAK_THRESHOLD]
        for _, r in sub.iterrows():
            model = r['model']
            g = clean[(clean['marka'] == marka) & (clean['model'] == model)]
            probe_err = probe_row_and_error(marka, model, clean, artifact, hp_lookup)
            eng_c = engine_consistency(g)
            body_c = body_type_consistency(g)
            gen_c = generation_consistency(r['price_age_corr'], r['price_cv'])
            safe = bool(r['synthetic_safe']) and 'HETEROJEN' not in eng_c
            n = int(r['train_real_count'])
            is_premium = r['median_price'] >= premium_threshold if pd.notna(r['median_price']) else False
            rec_n = recommend_count(n, safe)
            pr = priority(n, safe, probe_err, is_premium)

            reason_parts = [f'train_n={n}']
            if n <= 2:
                reason_parts.append('yetersiz demir (n<=2)')
            elif n > 20:
                reason_parts.append('zaten yeterli (n>20)')
            if 'HETEROJEN' in eng_c:
                reason_parts.append(f'motor heterojen: {eng_c}')
            if probe_err is not None:
                reason_parts.append(f'probe hata %{probe_err:.1f}')
            if is_premium:
                reason_parts.append('premium segment')

            rows.append({
                'marka': marka, 'model': model, 'train_real_count': n,
                'raw_real_count': int(r['raw_real_count']), 'hp_support': int(r['current_hp_support']),
                'hp_source': r['current_hp_source'],
                'unique_year_count': int(r['unique_year_count']),
                'year_range': f"{r['min_year']:.0f}-{r['max_year']:.0f}" if pd.notna(r['min_year']) else 'N/A',
                'km_range': f"{r['min_km']:,.0f}-{r['max_km']:,.0f}" if pd.notna(r['min_km']) else 'N/A',
                'min_price': r['min_price'], 'median_price': r['median_price'], 'max_price': r['max_price'],
                'arabam_count': int(r['arabam_count']), 'kaggle_count': int(r['kaggle_count']),
                'current_scrape_ratio': round(100 * r['arabam_count'] / n, 1) if n else 0.0,
                'current_prediction_probe_error_pct': round(probe_err, 1) if probe_err is not None else None,
                'synthetic_safe': safe,
                'generation_consistency': gen_c, 'engine_consistency': eng_c, 'body_type_consistency': body_c,
                'recommended_synthetic_count': rec_n, 'candidate_priority': pr,
                'reason': '; '.join(reason_parts),
            })
    return pd.DataFrame(rows)


def main():
    print('reports/synthetic_candidates_full.csv (Faz30, hala gecerli) okunuyor...')
    cand_df = pd.read_csv(CANDIDATES_PATH)

    print('production preprocessing + artifact okunuyor (SADECE OKUMA)...')
    clean = load_clean_train_dataset()
    artifact = joblib.load(MODEL_PATH)
    hp_lookup = artifact['hierarchical_price']

    premium_threshold = cand_df['median_price'].quantile(0.85)
    print(f'premium esigi (p85): {premium_threshold:,.0f}')

    primary_df = build_table(PRIMARY_BRANDS, False, cand_df, clean, artifact, hp_lookup, premium_threshold)
    secondary_df = build_table(SECONDARY_BRANDS, True, cand_df, clean, artifact, hp_lookup, premium_threshold)
    full = pd.concat([primary_df, secondary_df], ignore_index=True)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    full.to_csv(os.path.join(REPORTS_DIR, 'third_wave_candidates_full.csv'), index=False, encoding='utf-8-sig')

    for marka in PRIMARY_BRANDS + SECONDARY_BRANDS:
        sub = full[full['marka'] == marka].sort_values('train_real_count')
        print(f'\n=== {marka} ({len(sub)} model) ===')
        cols = ['model', 'train_real_count', 'raw_real_count', 'hp_support', 'unique_year_count',
                'median_price', 'arabam_count', 'current_prediction_probe_error_pct',
                'synthetic_safe', 'engine_consistency', 'candidate_priority', 'recommended_synthetic_count']
        print(sub[cols].to_string(index=False))

    print(f'\nToplam analiz edilen model: {len(full)}')
    print(f'Yazildi: {REPORTS_DIR}/third_wave_candidates_full.csv')


if __name__ == '__main__':
    main()
