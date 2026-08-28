"""Faz 31 - ikinci kontrollu sentetik veri pilot dalgasi icin ADAY SECIMI +
ANALIZ. SADECE ANALIZ - hicbir sentetik satir uretmez, train_dataset.csv'ye
dokunmaz, production artifact'i retrain etmez, mevcut synthetic_pilot.csv'yi
degistirmez.

VERI KAYNAGI: reports/synthetic_candidates_full.csv (Faz 30'da uretilen,
production preprocessing + guncel artifact'in hierarchical_price/hp_support'undan
turetilmis) - bu rapor hala GECERLI, cunku hierarchical_price/hp_support HER ZAMAN
sadece GERCEK egitim verisinden hesaplaniyor (bkz. Faz 30 mimarisi) ve pilot
sentetik (18 satir) bu sayilari hic degistirmedi - Faz 31'de bunu ayrica dogrular.

Calistirma (ai-model/ calisma dizini olarak): python analyze_second_wave_candidates.py
"""
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import CURRENT_YEAR, PRICE_REFERENCE_DATE, load_clean_train_dataset
from hp_support import lookup_support, compute_confidence
import hierarchical_price as hp
from train import CATEGORICAL_COLS, apply_saved_categories

BASE_DIR = os.path.dirname(__file__)
CANDIDATES_PATH = os.path.join(BASE_DIR, 'reports', 'synthetic_candidates_full.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

EXCLUDE_PILOT = {('Ferrari', '458'), ('Lamborghini', 'Huracan'), ('Rolls-Royce', 'Ghost')}
EXCLUDE_SUFFICIENT = {('Cadillac', 'Escalade'), ('Bentley', 'Continental')}

TOTAL_SYNTHETIC_CAP = 50
WAVE_TARGET_MIN, WAVE_TARGET_MAX = 5, 10

FIXED_CHECK_PATTERNS = [
    ('Aston Martin', None), ('Maserati', None), ('McLaren', None),
    ('Porsche', None), ('Mercedes - Benz', r'Maybach|S \d|AMG'),
    ('Land Rover', r'Range Rover'), ('Range Rover', None),
    ('BMW', r'^M\d|^M$|i7|XM'), ('Audi', r'^RS|e-tron GT|E-Tron GT'),
    ('Ferrari', None), ('Lamborghini', None), ('Rolls-Royce', None),
]


def _current_scrape_ratio(row):
    return row['arabam_count'] / row['raw_real_count'] if row['raw_real_count'] > 0 else 0.0


def _is_premium(row, premium_threshold):
    return row['median_price'] >= premium_threshold


def _score_and_reason(row, premium_threshold):
    score = 0
    reasons, risks = [], []
    n = row['train_real_count']

    if 3 <= n <= 7:
        score += 20; reasons.append(f'train support {n} (3-7 bandi, +20)')
    elif 8 <= n <= 12:
        score += 12; reasons.append(f'train support {n} (8-12 bandi, +12)')

    if row['synthetic_safe']:
        score += 15; reasons.append('synthetic_safe=true (+15)')

    if row['unique_year_count'] >= 3:
        score += 10; reasons.append(f'{row["unique_year_count"]} farkli yil (+10)')

    has_current_scrape = row['arabam_count'] >= 1
    if has_current_scrape:
        score += 15; reasons.append(f'{row["arabam_count"]} guncel arabam- kayit (+15)')

    is_premium = _is_premium(row, premium_threshold)
    if is_premium:
        score += 10; reasons.append(f'premium/high-value (medyan {row["median_price"]:,.0f} >= {premium_threshold:,.0f}) (+10)')

    low_support = row['current_hp_support'] < 10 or row['current_hp_source'] != 'brand_model'
    if low_support:
        score += 10; reasons.append(f'dusuk hp_support/zayif fallback ({row["current_hp_support"]}, {row["current_hp_source"]}) (+10)')

    km_span_ok = (row['max_km'] - row['min_km']) > 5000 if pd.notna(row['max_km']) and pd.notna(row['min_km']) else False
    year_span_ok = (row['max_year'] - row['min_year']) >= 2 if pd.notna(row['max_year']) else False
    interp_good = km_span_ok and year_span_ok
    if interp_good:
        score += 10; reasons.append('interpolasyon araligi (yil+km) yeterli genislikte (+10)')

    price_consistent = (row['price_age_corr'] is not None and not pd.isna(row['price_age_corr']) and row['price_age_corr'] <= -0.3) or \
                        (pd.notna(row['price_cv']) and row['price_cv'] <= 0.5)
    if price_consistent:
        score += 10; reasons.append('fiyat davranisi tutarli (yasla aciklanan varyans veya dusuk cv) (+10)')

    only_legacy = row['arabam_count'] == 0 and row['kaggle_count'] > 0
    if only_legacy:
        score -= 15; risks.append('sadece Kaggle/legacy kaynak, guncel scrape yok (-15)')

    if n <= 2:
        score -= 25; risks.append('train n<=2, yetersiz demir (-25)')

    if row['unique_year_count'] <= 1:
        score -= 20; risks.append('tek model yili (-20)')

    if not row['synthetic_safe'] and 'fiyat varyasyonu asiri yuksek' in str(row['safety_reason']):
        score -= 20; risks.append('price_cv asiri yuksek, yasla aciklanamiyor (-20)')

    if not interp_good:
        score -= 15; risks.append('interpolasyon alani dar (yil veya km araligi cok kisitli) (-15)')

    if pd.notna(row['max_year']) and row['max_year'] < (CURRENT_YEAR - 15) and row['raw_real_count'] <= 8:
        score -= 10; risks.append(f'cok eski model (max_year={row["max_year"]:.0f}), dusuk ticari onem (-10)')

    if not row['in_ui_catalog']:
        score -= 10; risks.append('UI kataloginda secilebilir DEGIL (-10)')

    score = max(0, min(100, score))
    return score, '; '.join(reasons), '; '.join(risks) if risks else 'yok'


def _proposed_synthetic_count(n, score, safe):
    if n <= 2 or not safe:
        return 0
    if 3 <= n <= 5:
        return 6 if score >= 70 else 4
    if 6 <= n <= 8:
        return 6 if score >= 70 else 4
    if 9 <= n <= 12:
        return 5 if score >= 70 else 3
    if n > 12:
        return 3 if score >= 80 else 0
    return 0


def build_probe_row(marka, model, clean_df, X_train_cols):
    g = clean_df[(clean_df['marka'] == marka) & (clean_df['model'] == model)]
    if len(g) == 0:
        return None, None
    # temsili satir motor_gucu'su dolu OLMALI (hp_support confidence lookup'u
    # icin gerekli) - Faz 30 sonrasi bazi gercek satirlarda NaN olabilir.
    g_hp_ok = g[g['motor_gucu'].notna()]
    g_for_pick = g_hp_ok if len(g_hp_ok) > 0 else g
    med_idx = (g_for_pick['fiyat'] - g_for_pick['fiyat'].median()).abs().idxmin()
    rep = g_for_pick.loc[med_idx]
    row = pd.DataFrame([rep]).drop(columns=['fiyat', 'ilan_id'], errors='ignore')
    row = row.reindex(columns=X_train_cols)
    return row, rep['fiyat']


def main():
    print('reports/synthetic_candidates_full.csv okunuyor (Faz 30, gecerliligini koruyor)...')
    df = pd.read_csv(CANDIDATES_PATH)
    df['reason'] = df['reason'].astype(str)
    df['safety_reason'] = df['safety_reason'].astype(str)

    print('production artifact okunuyor (SADECE OKUMA - hp_support/hierarchical_price real-only dogrulanacak)...')
    artifact = joblib.load(MODEL_PATH)
    hp_support_art = artifact['hp_support']
    hp_lookup = artifact['hierarchical_price']
    model = artifact['model']

    # dogrulama: pilot 3 grubun hp_support/hierarchical_price'i HALA real-only mi
    for marka, model_name in EXCLUDE_PILOT:
        key = f'{marka}\x1f{model_name}'
        c = hp_lookup['brand_model_curve'].get(key)
        s = hp_support_art['model_stats'].get(key)
        print(f'  dogrulama {marka} {model_name}: hierarchical_price n={c["n"] if c else None}, hp_support count={s["count"] if s else None}')

    print('\nproduction preprocessing ile temiz veri okunuyor (probe icin - SADECE OKUMA)...')
    clean = load_clean_train_dataset()

    exclude_keys = EXCLUDE_PILOT | EXCLUDE_SUFFICIENT
    df['exclude'] = df.apply(lambda r: (r['marka'], r['model']) in exclude_keys, axis=1)
    pool = df[~df['exclude']].copy()
    print(f'\nHavuz (pilot 3 grup + Escalade/Bentley Continental cikarilmis): {len(pool)} grup')

    premium_threshold = df['median_price'].quantile(0.85)
    print(f'premium esigi (p85, veri-turevli): {premium_threshold:,.0f} TL')

    # --- reject/no-need kategorileri ---
    insufficient_anchors = pool[pool['train_real_count'] <= 2].copy()
    unsafe = pool[(pool['train_real_count'] > 2) & (~pool['synthetic_safe'])].copy()
    legacy_only = pool[(pool['train_real_count'] > 2) & (pool['synthetic_safe']) &
                        (pool['arabam_count'] == 0) & (pool['kaggle_count'] > 0)].copy()
    no_need = pool[pool['train_real_count'] > 12].copy()

    # --- aday havuzu: kriter 2/3'u saglayan (n 3-12 araliginda, guvenli, en az 2 yil) ---
    eligible = pool[
        (pool['train_real_count'] >= 3) & (pool['train_real_count'] <= 12) &
        (pool['synthetic_safe']) & (pool['unique_year_count'] >= 2)
    ].copy()
    print(f'Uygun aday havuzu (n 3-12, safe, >=2 yil): {len(eligible)} grup')

    scores, reasons, risks, proposed = [], [], [], []
    for _, row in eligible.iterrows():
        s, r, rk = _score_and_reason(row, premium_threshold)
        scores.append(s); reasons.append(r); risks.append(rk)
        proposed.append(_proposed_synthetic_count(row['train_real_count'], s, row['synthetic_safe']))
    eligible['second_wave_score'] = scores
    eligible['reason'] = reasons
    eligible['risk_notes'] = risks
    eligible['proposed_synthetic_count'] = proposed
    eligible['current_scrape_ratio'] = eligible.apply(_current_scrape_ratio, axis=1)

    top20 = eligible.sort_values('second_wave_score', ascending=False).head(20).copy()

    print('\n=== TOP 20 ADAY - PROBE (mevcut production model) ===')
    probe_rows = []
    for _, r in top20.iterrows():
        marka, model_name = r['marka'], r['model']
        row_df, actual = build_probe_row(marka, model_name, clean, list(artifact['feature_columns']))
        if row_df is None:
            probe_rows.append({'marka': marka, 'model': model_name, 'actual': None, 'pred': None,
                                'abs_err': None, 'pct_err': None, 'confidence': None, 'hp_support_conf': None})
            continue
        for c in CATEGORICAL_COLS:
            row_df[c] = row_df[c].astype('category').cat.set_categories(artifact['category_sets'][c])
        yas = row_df['yas'].iloc[0]
        hp_val, hp_src, hp_n = hp.lookup_price(marka, model_name, yas, hp_lookup)
        row_df[hp.FEATURE_COLUMN] = hp_val
        row_aligned = apply_saved_categories(row_df, artifact)
        pred = float(model.predict(row_aligned)[0])
        enginepower = row_df['motor_gucu'].iloc[0]
        if pd.notna(enginepower):
            peer_count, model_count, pct, peer_group = lookup_support(marka, model_name, enginepower, hp_support_art)
            conf = compute_confidence(peer_count, model_count, peer_group)
        else:
            # temsili satirin motor_gucu'su HAM veride de NaN (Faz 30 native-missing
            # birakma kurali) - HP-confidence bin'i motor_gucu gerektirir, hesaplanamaz.
            model_count, conf = None, 'n/a (motor_gucu bilinmiyor)'
        abs_err = abs(pred - actual)
        pct_err = 100 * abs_err / actual
        probe_rows.append({'marka': marka, 'model': model_name, 'actual': actual, 'pred': pred,
                            'abs_err': abs_err, 'pct_err': pct_err, 'confidence': conf, 'hp_support_conf': model_count})
        print(f'  {marka} {model_name}: actual={actual:,.0f} pred={pred:,.0f} abs_err={abs_err:,.0f} '
              f'pct_err={pct_err:.1f}% confidence={conf} hp_support={model_count}')

    probe_df = pd.DataFrame(probe_rows)
    top20 = top20.merge(probe_df, on=['marka', 'model'], how='left')

    # yuksek hata riski dusukse (model zaten iyi tahmin ediyorsa) skoru hafifce indir
    top20['second_wave_score'] = top20.apply(
        lambda r: max(0, r['second_wave_score'] - 10) if (r['pct_err'] is not None and r['pct_err'] < 8) else r['second_wave_score'],
        axis=1,
    )
    top20 = top20.sort_values('second_wave_score', ascending=False)

    # --- RECOMMENDED: cesitlilik + toplam sentetik <= TOTAL_SYNTHETIC_CAP ---
    recommended = []
    used_brands = {}
    total_synth = 0
    for _, r in top20.iterrows():
        if len(recommended) >= WAVE_TARGET_MAX:
            break
        brand_count = used_brands.get(r['marka'], 0)
        if brand_count >= 2:  # kural 11: tek markaya yigilma
            continue
        if r['proposed_synthetic_count'] == 0:
            continue
        if total_synth + r['proposed_synthetic_count'] > TOTAL_SYNTHETIC_CAP:
            continue
        recommended.append(r)
        used_brands[r['marka']] = brand_count + 1
        total_synth += r['proposed_synthetic_count']
    if len(recommended) < WAVE_TARGET_MIN:
        print(f'\nUYARI: cesitlilik/sinir kisitlariyla sadece {len(recommended)} model secilebildi (hedef {WAVE_TARGET_MIN}-{WAVE_TARGET_MAX})')
    recommended_df = pd.DataFrame(recommended)

    print(f'\n=== RECOMMENDED SECOND-WAVE PILOT ({len(recommended_df)} model, toplam {total_synth} sentetik) ===')
    print(recommended_df[['marka', 'model', 'train_real_count', 'second_wave_score', 'proposed_synthetic_count']].to_string(index=False))

    print('\n=== SABIT KONTROL LISTESI ===')
    fixed_hits = []
    for marka_pat, model_pat in FIXED_CHECK_PATTERNS:
        matches = df[df['marka'] == marka_pat]
        if model_pat:
            matches = matches[matches['model'].astype(str).str.contains(model_pat, regex=True, na=False)]
        for _, m in matches.iterrows():
            fixed_hits.append(m)
    if fixed_hits:
        fixed_df = pd.DataFrame(fixed_hits).drop_duplicates(subset=['marka', 'model'])
        print(fixed_df[['marka', 'model', 'raw_real_count', 'train_real_count', 'median_price', 'synthetic_safe', 'priority']].to_string(index=False))
    else:
        print('  (eslesme yok)')

    os.makedirs(REPORTS_DIR, exist_ok=True)
    eligible.sort_values('second_wave_score', ascending=False).to_csv(
        os.path.join(REPORTS_DIR, 'second_wave_candidates_full.csv'), index=False, encoding='utf-8-sig')
    top20.to_csv(os.path.join(REPORTS_DIR, 'second_wave_candidates_top20.csv'), index=False, encoding='utf-8-sig')

    print('\n=== OZET ===')
    print(f'Havuz: {len(pool)} grup')
    print(f'Uygun aday: {len(eligible)}')
    print(f'Insufficient anchors (n<=2): {len(insufficient_anchors)}')
    print(f'Unsafe (synthetic_safe=false, n>2): {len(unsafe)}')
    print(f'Legacy-only (arabam=0, n>2, safe): {len(legacy_only)}')
    print(f'No-need (n>12): {len(no_need)}')
    print(f'Top20 -> Recommended: {len(recommended_df)}, toplam onerilen sentetik: {total_synth}')

    with open(os.path.join(REPORTS_DIR, 'second_wave_recommendation.md'), 'w', encoding='utf-8') as f:
        f.write('# Faz 31 - Ikinci Dalga Sentetik Aday Analizi\n\n')
        f.write(f'Havuz: {len(pool)} grup (pilot 3 + Escalade/Bentley cikarilmis)\n\n')
        f.write(f'Premium esigi (p85): {premium_threshold:,.0f} TL\n\n')
        f.write('## TOP 20\n\n')
        f.write(top20[['marka', 'model', 'train_real_count', 'current_hp_support', 'min_year', 'max_year',
                        'min_km', 'max_km', 'median_price', 'arabam_count', 'second_wave_score',
                        'proposed_synthetic_count', 'risk_notes']].to_markdown(index=False))
        f.write('\n\n## RECOMMENDED SECOND-WAVE PILOT\n\n')
        f.write(recommended_df[['marka', 'model', 'train_real_count', 'second_wave_score',
                                 'proposed_synthetic_count', 'reason']].to_markdown(index=False))
        f.write(f'\n\nToplam onerilen sentetik: {total_synth}\n')

    print(f'\nRaporlar yazildi: {REPORTS_DIR}')


if __name__ == '__main__':
    main()
