"""Faz 33 - marka bazli GERCEK egitim verisi envanteri. SADECE ANALIZ - hicbir
sentetik veri uretmez, train_dataset.csv'ye dokunmaz, retrain/artifact
degisikligi YAPMAZ.

Veri kaynagi: preprocess.load_clean_train_dataset() (production preprocessing -
marka-ici q99 + km filtresi + REQUIRED_COLS + kategorik "Belirtilmemis" +
motor_hacmi/motor_gucu native missing) - SENTETIK SATIRLAR bu fonksiyona HIC
girmez (sadece train_dataset.csv okur), bu yuzden rapor otomatik olarak
sentetik-haric gercek veriyi yansitir.

Calistirma (ai-model/ calisma dizini olarak): python analyze_brand_inventory.py
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import CURRENT_YEAR, TRAIN_PATH, load_clean_train_dataset

BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
WAVE30_PATH = os.path.join(BASE_DIR, '..', 'data', 'output', 'synthetic_pilot.csv')
WAVE31_PATH = os.path.join(BASE_DIR, '..', 'data', 'output', 'synthetic_second_wave_preview.csv')

SPECIAL_CHECK_BRANDS = ['Honda', 'Toyota', 'Renault', 'Volkswagen', 'BMW', 'Mercedes - Benz', 'Audi',
                         'Porsche', 'Ferrari', 'Lamborghini', 'Rolls-Royce', 'Bentley', 'Aston Martin',
                         'Maserati', 'McLaren', 'Cadillac', 'Lexus', 'Dodge']

SUPPORT_BUCKETS = [
    ('CRITICAL', 1, 5), ('VERY_LOW', 6, 10), ('LOW', 11, 25), ('MODERATE', 26, 50),
    ('MEDIUM', 51, 100), ('GOOD', 101, 500), ('STRONG', 501, 2000), ('VERY_STRONG', 2001, float('inf')),
]


def tl(x):
    if pd.isna(x):
        return '-'
    return f'{x:,.0f} TL'.replace(',', '.')


def support_level(n):
    for label, lo, hi in SUPPORT_BUCKETS:
        if lo <= n <= hi:
            return label
    return 'NONE'


def _source_label(ilan_id):
    s = str(ilan_id)
    if s.startswith('arabam-'):
        return 'arabam'
    if s.startswith('kaggle-'):
        return 'kaggle'
    return 'other'


def main():
    print('Ham veri okunuyor...')
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw_valid = raw.dropna(subset=['marka', 'model']).copy()
    raw_valid['marka'] = raw_valid['marka'].astype(str).str.strip()

    print('Production preprocessing (SADECE OKUMA, sentetik HARIC)...')
    clean = load_clean_train_dataset()
    clean['_source'] = clean['ilan_id'].map(_source_label)

    print('Sentetik dalgalar (marka-seviyesi kullanim tespiti icin) okunuyor...')
    wave30 = pd.read_csv(WAVE30_PATH) if os.path.exists(WAVE30_PATH) else pd.DataFrame(columns=['marka', 'model'])
    wave31 = pd.read_csv(WAVE31_PATH) if os.path.exists(WAVE31_PATH) else pd.DataFrame(columns=['marka', 'model'])
    synth_all = pd.concat([wave30[['marka', 'model']], wave31[['marka', 'model']]], ignore_index=True)
    synth_by_brand_model = synth_all.groupby(['marka', 'model']).size()
    synth_by_brand = synth_all.groupby('marka').size()

    brand_model_train_counts = clean.groupby(['marka', 'model'], observed=True).size()

    raw_counts = raw_valid.groupby('marka').size()
    brands = sorted(raw_counts.index)

    rows = []
    for marka in brands:
        raw_n = int(raw_counts.get(marka, 0))
        g_train = clean[clean['marka'] == marka]
        train_n = len(g_train)
        if train_n == 0:
            models_n = years_n = 0
            min_yil = max_yil = None
            min_p = med_p = mean_p = max_p = p10 = p90 = None
            min_km = med_km = max_km = None
            arabam_n = kaggle_n = 0
            weak_count = 0
        else:
            models_n = g_train['model'].nunique()
            years_n = g_train['yil'].nunique()
            min_yil, max_yil = g_train['yil'].min(), g_train['yil'].max()
            prices = g_train['fiyat']
            min_p, med_p, mean_p, max_p = prices.min(), prices.median(), prices.mean(), prices.max()
            p10, p90 = prices.quantile(0.10), prices.quantile(0.90)
            kms = g_train['kilometre']
            min_km, med_km, max_km = kms.min(), kms.median(), kms.max()
            src_counts = g_train['_source'].value_counts()
            arabam_n = int(src_counts.get('arabam', 0))
            kaggle_n = int(src_counts.get('kaggle', 0))
            brand_models = g_train['model'].unique()
            bm_counts = [brand_model_train_counts.get((marka, m), 0) for m in brand_models]
            weak_count = sum(1 for c in bm_counts if c < 10)

        n_le2 = n_35 = n_610 = n_1120 = n_gt20 = 0
        if train_n > 0:
            for m in g_train['model'].unique():
                c = brand_model_train_counts.get((marka, m), 0)
                if c <= 2:
                    n_le2 += 1
                elif c <= 5:
                    n_35 += 1
                elif c <= 10:
                    n_610 += 1
                elif c <= 20:
                    n_1120 += 1
                else:
                    n_gt20 += 1

        synth_used = marka in synth_by_brand.index
        synth_count = int(synth_by_brand.get(marka, 0))

        rows.append({
            'marka': marka, 'raw_ilan_sayisi': raw_n, 'train_ilan_sayisi': train_n,
            'kayip_ilan_sayisi': raw_n - train_n,
            'retention_pct': round(100 * train_n / raw_n, 1) if raw_n else 0.0,
            'benzersiz_model_sayisi': int(models_n), 'benzersiz_yil_sayisi': int(years_n),
            'min_yil': min_yil, 'max_yil': max_yil,
            'min_fiyat': min_p, 'median_fiyat': med_p, 'mean_fiyat': mean_p, 'max_fiyat': max_p,
            'fiyat_araligi': (max_p - min_p) if train_n else None,
            'p10_fiyat': p10, 'p90_fiyat': p90,
            'min_km': min_km, 'median_km': med_km, 'max_km': max_km,
            'arabam_ilan_sayisi': arabam_n, 'kaggle_legacy_ilan_sayisi': kaggle_n,
            'current_scrape_ratio': round(100 * arabam_n / train_n, 1) if train_n else 0.0,
            'synthetic_currently_used': synth_used, 'current_synthetic_count': synth_count,
            'brand_support_level': support_level(train_n) if train_n > 0 else 'NONE',
            'models_with_n_le_2': n_le2, 'models_with_n_3_5': n_35, 'models_with_n_6_10': n_610,
            'models_with_n_11_20': n_1120, 'models_with_n_gt_20': n_gt20,
            'weak_model_count': weak_count,
            'weak_model_ratio': round(100 * weak_count / models_n, 1) if train_n and models_n else None,
        })

    df = pd.DataFrame(rows)

    # premium esigi: marka MEDYAN fiyatlarinin dagilimindaki p90 (veri-turevli, hardcode YOK)
    valid_med = df[df['train_ilan_sayisi'] > 0]['median_fiyat']
    premium_threshold = valid_med.quantile(0.90)

    def priority_and_reason(row):
        if row['train_ilan_sayisi'] == 0:
            return 'NONE', False, f"{row['marka']}: bu markadan production egitiminde hic gercek satir yok (preprocessing sonrasi 0)."
        n = row['train_ilan_sayisi']
        is_premium = row['median_fiyat'] is not None and row['median_fiyat'] >= premium_threshold
        weak_ratio = row['weak_model_ratio'] or 0
        score = 0
        if n < 10: score += 3
        elif n < 50: score += 2
        elif n < 200: score += 1
        if is_premium: score += 2
        if weak_ratio >= 70: score += 2
        elif weak_ratio >= 40: score += 1
        if row['current_scrape_ratio'] < 20: score += 1

        if score >= 6: pr = 'CRITICAL'
        elif score >= 4: pr = 'HIGH'
        elif score >= 2: pr = 'MEDIUM'
        elif score >= 1: pr = 'LOW'
        else: pr = 'NONE'
        candidate = pr in ('CRITICAL', 'HIGH', 'MEDIUM')

        if candidate:
            reason = (f"Toplam gercek kayit {'dusuk' if n < 50 else 'orta'} ({n}), "
                      f"{'yuksek fiyat segmenti, ' if is_premium else ''}"
                      f"{weak_ratio:.0f}% model dusuk destekli (n<10).")
        else:
            reason = 'Yeterli gercek kayit; marka duzeyinde sentetik destege ihtiyac yok.'
        return pr, candidate, reason

    priorities, candidates, reasons = [], [], []
    for _, row in df.iterrows():
        pr, cand, reason = priority_and_reason(row)
        priorities.append(pr); candidates.append(cand); reasons.append(reason)
    df['synthetic_priority'] = priorities
    df['preliminary_synthetic_candidate'] = candidates
    df['synthetic_reason'] = reasons

    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.sort_values('train_ilan_sayisi', ascending=False).to_csv(
        os.path.join(REPORTS_DIR, 'brand_inventory_full.csv'), index=False, encoding='utf-8-sig')

    # ==================== SUMMARY.MD ====================
    lines = []
    lines.append('# Faz 33 - Marka Bazli Egitim Verisi Envanteri\n')
    valid_df = df[df['train_ilan_sayisi'] > 0]
    lines.append(f'Toplam marka: **{len(df)}** ({len(valid_df)} tanesinde en az 1 egitim satiri)\n')
    lines.append(f'Toplam gercek train ilan: **{int(df["train_ilan_sayisi"].sum()):,}**\n'.replace(',', '.'))
    lines.append(f'Medyan marka ilan sayisi: **{int(valid_df["train_ilan_sayisi"].median()):,}**\n'.replace(',', '.'))
    top_brand = valid_df.loc[valid_df['train_ilan_sayisi'].idxmax()]
    bot_brand = valid_df.loc[valid_df['train_ilan_sayisi'].idxmin()]
    lines.append(f'En cok verili marka: **{top_brand["marka"]}** ({int(top_brand["train_ilan_sayisi"]):,})\n'.replace(',', '.'))
    lines.append(f'En az verili marka: **{bot_brand["marka"]}** ({int(bot_brand["train_ilan_sayisi"])})\n')
    premium_thin = valid_df[(valid_df['median_fiyat'] >= premium_threshold) & (valid_df['train_ilan_sayisi'] < 100)]
    lines.append(f'Premium+az verili marka sayisi (medyan>=p90 VE n<100): **{len(premium_thin)}**\n')
    lines.append(f'\nPremium esigi (marka medyan fiyat dagiliminin p90\'i, veri-turevli): **{tl(premium_threshold)}**\n')

    def fmt_row_a(r):
        return (f"| {r['marka']} | {int(r['train_ilan_sayisi']):,} | {int(r['benzersiz_model_sayisi'])} | "
                f"{tl(r['min_fiyat'])} | {tl(r['median_fiyat'])} | {tl(r['max_fiyat'])} | "
                f"{int(r['arabam_ilan_sayisi'])} | {r['weak_model_count']} | {r['weak_model_ratio']}% |").replace(',', '.')

    lines.append('\n## TABLO A — Tum markalar (ilan sayisina gore)\n')
    lines.append('| Marka | Train ilan | Model | Min fiyat | Medyan | Max fiyat | Guncel scrape | Weak model | Weak % |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for _, r in valid_df.sort_values('train_ilan_sayisi', ascending=False).iterrows():
        lines.append(fmt_row_a(r))

    # TABLO B - birlesik skor: dusuk n + yuksek weak_ratio + premium + dusuk scrape orani
    scored = valid_df.copy()
    scored['_score'] = (
        (1 / (scored['train_ilan_sayisi'] + 1)) * 1000 +
        scored['weak_model_ratio'].fillna(0) * 2 +
        (scored['median_fiyat'] >= premium_threshold).astype(int) * 500 +
        (100 - scored['current_scrape_ratio']) * 0.5
    )
    top30 = scored.sort_values('_score', ascending=False).head(30)
    lines.append('\n## TABLO B — Sentetik veri acisindan en dikkat edilmesi gereken 30 marka\n')
    lines.append('| Rank | Marka | Train ilan | Model | Weak model | Weak % | Min fiyat | Medyan fiyat | Max fiyat | Guncel scrape | Priority |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for i, (_, r) in enumerate(top30.iterrows(), 1):
        lines.append(f"| {i} | {r['marka']} | {int(r['train_ilan_sayisi']):,} | {int(r['benzersiz_model_sayisi'])} | "
                      f"{r['weak_model_count']} | {r['weak_model_ratio']}% | {tl(r['min_fiyat'])} | {tl(r['median_fiyat'])} | "
                      f"{tl(r['max_fiyat'])} | {int(r['arabam_ilan_sayisi'])} | {r['synthetic_priority']} |".replace(',', '.'))

    lines.append('\n## OZEL KONTROL — belirlenen markalar\n')
    lines.append('| Marka | Train ilan | Model | Min fiyat | Medyan | Max fiyat | Weak model | Weak % | Support level | Synthetic var mi |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for b in SPECIAL_CHECK_BRANDS:
        match = df[df['marka'] == b]
        if len(match) == 0:
            lines.append(f'| {b} | - | - | - | - | - | - | - | - | dataset\'te yok |')
            continue
        r = match.iloc[0]
        lines.append(f"| {b} | {int(r['train_ilan_sayisi']):,} | {int(r['benzersiz_model_sayisi'])} | "
                      f"{tl(r['min_fiyat'])} | {tl(r['median_fiyat'])} | {tl(r['max_fiyat'])} | "
                      f"{r['weak_model_count']} | {r['weak_model_ratio']}% | {r['brand_support_level']} | "
                      f"{'Evet' if r['synthetic_currently_used'] else 'Hayir'} |".replace(',', '.'))

    lines.append('\n---\n**ONEMLI NOT:** Marka bazli dusuk support TEK BASINA sentetik veri uretme karari DEGILDIR. '
                  'Nihai sentetik uretim brand_model seviyesindeki gercek kayit, yil/km dagilimi, generation '
                  'tutarliligi ve interpolation guvenligi incelendikten sonra yapilmalidir.\n')

    with open(os.path.join(REPORTS_DIR, 'brand_inventory_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'\nToplam marka: {len(df)}, gecerli (n>0): {len(valid_df)}')
    print(f'Toplam train ilan: {int(df["train_ilan_sayisi"].sum()):,}')
    print(f'Premium esigi (p90 marka medyani): {premium_threshold:,.0f}')
    print(f'Premium+az verili (n<100): {len(premium_thin)}')
    print(f'Raporlar yazildi: {REPORTS_DIR}')


if __name__ == '__main__':
    main()
