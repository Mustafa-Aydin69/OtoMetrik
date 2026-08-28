"""Faz 30: kontrollu sentetik veri uretimine ADAY marka+model gruplarini tespit
eden, tekrar calistirilabilir, SADECE ANALIZ scripti.

BU SCRIPT: train_dataset.csv'yi, models/lightgbm_final.joblib'i DEGISTIRMEZ, hicbir
sentetik kayit URETMEZ, retrain YAPMAZ - sadece mevcut gercek veriyi + production
hierarchical_price artefaktini okuyup reports/ altina CSV/MD rapor yazar.

VERI KAYNAKLARI:
- raw_real_count: train_dataset.csv'nin HAM hali (preprocess.py hic uygulanmamis).
- train_real_count: preprocess.load_clean_train_dataset() SONRASI (Faz 29 marka-ici
  q99 + km<=1M + dropna) - production egitiminin GERCEKTEN gordugu satir sayisi.
  Bu iki sayi arasindaki fark, preprocessing'in o grup icin ne kadar veri kaybına
  yol actigini gosterir (bkz. TABLO 1/4 - raw_real_count >> train_real_count olan
  gruplar ozel ilgi gerektirir).
- current_hp_support / current_hp_method: models/lightgbm_final.joblib icindeki
  'hierarchical_price' lookup'u (bkz. hierarchical_price.py, Faz 29) - HANGI
  katmanin (brand_model/model/brand/global) bu grup icin kullanilacagini VE o
  katmanin kac satirdan fit edildigini (curve['n']) okur - egitim/retrain YAPMAZ,
  sadece diskteki artefakti okur.

Faz 29'un kok neden analizinden (Ferrari/Lamborghini vakasi) hatirlatma: ESKI
global q99 kirpmasi premium markalari TAMAMEN siliyordu - bu artik duzeltildi
(preprocess.py marka-ici q99), yani train_real_count artik raw_real_count'a
COK daha yakin olmali premium markalarda da. Bu script "kirpma yuzunden mi az"
sorusunu (raw vs train farkiyla) AYRI olarak "gercekten az ilan var mi" sorusundan
(raw_real_count'un kendisi) ayirt eder.

PREMIUM/DUSUK-FIYAT ESIKLERI: brand isimleri HARDCODE EDILMEZ (kullanici talebi) -
tum brand_model gruplarinin KENDI medyan fiyatlarinin dagilimindan turetilen
percentile esikleri kullanilir (bkz. main() - PREMIUM_PERCENTILE/LOW_PRICE_PERCENTILE).

Calistirma (ai-model/ calisma dizini olarak):
    python analyze_synthetic_data_candidates.py
"""
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import CURRENT_YEAR, TRAIN_PATH, load_clean_train_dataset

BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')
CATALOG_TS_PATH = os.path.join(BASE_DIR, '..', '..', 'WebSite', 'src', 'lib', 'vehicle-options.generated.ts')

# --- adaylik esikleri (kullanicinin gorev talebindeki taban mantik) ---
COUNT_BUCKETS = [
    ('CRITICAL', 0, 2),
    ('HIGH', 3, 5),
    ('MEDIUM', 6, 9),
    ('LOW', 10, 19),
    ('NONE', 20, float('inf')),
]
PRIORITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE']  # 0=en acil

PREMIUM_PERCENTILE = 0.90   # brand_model medyan-fiyat dagiliminin ustten %10'u -> "premium segment"
LOW_PRICE_PERCENTILE = 0.25  # alttan %25 -> "dusuk fiyat etkisi" (onceligi azaltan faktor)
FEW_YEARS_THRESHOLD = 2
WIDE_PRICE_RATIO = 3.0       # max_price/min_price bu esigin ustundeyse "cok genis aralik"
HIGH_CV_UNSAFE = 0.85        # bu ustunde price_cv -> veri ici tutarsizlik supheli, synthetic_safe=False
LOW_SUPPORT_THRESHOLD = 10   # current_hp_support bu altindaysa "dusuk destek"
OLD_STALE_MAX_AGE = 15       # max_year, CURRENT_YEAR'dan bu kadar eskiyse + raw kucukse "artik nadir piyasada"
FRESH_ARABAM_SUFFICIENT = 10  # arabam_count (guncel scrape) bu ustundeyse "zaten yeterli guncel ornek"

SPECIAL_CHECK_MODELS = [
    ('Ferrari', '458'),
    ('Lamborghini', 'Huracan'),
    ('Cadillac', 'Escalade'),
    ('Bentley', 'Continental'),
    ('Rolls-Royce', 'Ghost'),
]


def _ilan_source(ilan_id):
    s = str(ilan_id)
    if s.startswith('arabam-'):
        return 'arabam'
    if s.startswith('kaggle-ab-'):
        return 'kaggle_ab'
    if s.startswith('kaggle-ar-'):
        return 'kaggle_ar'
    return 'other'


def _cv(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.mean() == 0:
        return 0.0
    return float(values.std(ddof=0) / values.mean())


def _load_catalog_models():
    """vehicle-options.generated.ts'teki MODELS_BY_BRAND'i okur - bu grup UI
    dropdown'unda gercekten secilebilir mi kontrolu icin (retrain/pipeline
    calistirmadan, sadece mevcut uretilmis dosyayi parse eder)."""
    try:
        text = open(CATALOG_TS_PATH, encoding='utf-8').read()
    except OSError:
        return set()
    m = re.search(r'export const MODELS_BY_BRAND[^=]*= (\{.*?\n\});\n', text, re.S)
    if not m:
        return set()
    models_by_brand = json.loads(m.group(1))
    return {(marka, model) for marka, models in models_by_brand.items() for model in models}


def _resolve_hierarchical(marka, model, hp_lookup):
    """hierarchical_price.py'nin lookup_price() ile AYNI katman secim mantigi -
    ama sadece OKUMA, yas parametresi gerektirmez (method siniflandirmasi yas'tan
    bagimsiz, curve'un kendi slope'una gore belirlenir)."""
    key = f'{marka}\x1f{model}'
    bm = hp_lookup['brand_model_curve']
    if key in bm:
        c = bm[key]
        method = 'theil_sen' if c['slope'] != 0 else 'median'
        return 'brand_model', method, c['n']
    mm = hp_lookup.get('model_curve', {})
    if str(model) in mm:
        return 'model', 'fallback_model', mm[str(model)]['n']
    bc = hp_lookup['brand_curve']
    if marka in bc:
        return 'brand', 'fallback_brand', bc[marka]['n']
    gc = hp_lookup['global_curve']
    return 'global', 'fallback_global', gc['n']


def build_group_table(raw_df, clean_df, hp_lookup, catalog_models):
    raw_valid = raw_df.dropna(subset=['marka', 'model']).copy()
    raw_valid['marka'] = raw_valid['marka'].astype(str).str.strip()
    raw_valid['model'] = raw_valid['model'].astype(str).str.strip()
    raw_valid['_source'] = raw_valid['ilan_id'].map(_ilan_source)

    raw_counts = raw_valid.groupby(['marka', 'model'], observed=True).size().rename('raw_real_count')
    train_counts = clean_df.groupby(['marka', 'model'], observed=True).size().rename('train_real_count')

    rows = []
    for (marka, model), raw_count in raw_counts.items():
        train_count = int(train_counts.get((marka, model), 0))
        g_raw = raw_valid[(raw_valid['marka'] == marka) & (raw_valid['model'] == model)]
        g_clean = clean_df[(clean_df['marka'] == marka) & (clean_df['model'] == model)]

        src_counts = g_raw['_source'].value_counts().to_dict()
        arabam_n = int(src_counts.get('arabam', 0))
        kaggle_n = int(src_counts.get('kaggle_ab', 0) + src_counts.get('kaggle_ar', 0))
        other_n = int(src_counts.get('other', 0))
        source_distribution = ', '.join(
            f'{k}={v}' for k, v in [('arabam', arabam_n), ('kaggle', kaggle_n), ('other', other_n)] if v > 0
        ) or 'yok'

        # istatistikler: PRODUCTION EGITIMININ GORDUGU (temizlenmis) veriden -
        # sentetik interpolasyon araligi da bu "guvenilir" bolgeye dayanmali.
        # train_real_count==0 ise (preprocessing hepsini elemis) HAM satirlara
        # duser - bunlarda 'yas' preprocess.py'de hic uretilmedigi icin burada
        # ayni formulle (CURRENT_YEAR - yil) turetilir, sadece bu analiz icin.
        if len(g_clean) > 0:
            stat_source = g_clean
        else:
            stat_source = g_raw.assign(yas=CURRENT_YEAR - g_raw['yil'])
        years = stat_source['yil'].dropna()
        kms = stat_source['kilometre'].dropna()
        prices = stat_source['fiyat'].dropna()

        unique_year_count = int(years.nunique())
        min_year = float(years.min()) if len(years) else None
        max_year = float(years.max()) if len(years) else None
        min_km = float(kms.min()) if len(kms) else None
        median_km = float(kms.median()) if len(kms) else None
        max_km = float(kms.max()) if len(kms) else None
        min_price = float(prices.min()) if len(prices) else None
        median_price = float(prices.median()) if len(prices) else None
        max_price = float(prices.max()) if len(prices) else None
        price_cv = _cv(prices.values) if len(prices) else 0.0

        # yas-fiyat korelasyonu: yuksek price_cv YAS ile aciklanabiliyorsa (guclu
        # negatif korelasyon = normal amortisman, orn. Cadillac Escalade 2003-2023)
        # bu "veri tutarsizligi" DEGIL beklenen bir dagilimdir - safety kontrolunde
        # ayirt etmek icin kullanilir (bkz. classify_and_recommend).
        pair = stat_source[['yas', 'fiyat']].dropna()
        if len(pair) >= 3 and pair['yas'].nunique() >= 2 and (pair['fiyat'] > 0).all():
            price_age_corr = float(np.corrcoef(pair['yas'], np.log(pair['fiyat']))[0, 1])
        else:
            price_age_corr = None

        hp_source, hp_method, hp_support = _resolve_hierarchical(marka, model, hp_lookup)
        in_catalog = (marka, model) in catalog_models

        rows.append({
            'marka': marka, 'model': model,
            'raw_real_count': int(raw_count), 'train_real_count': train_count,
            'unique_year_count': unique_year_count, 'min_year': min_year, 'max_year': max_year,
            'min_km': min_km, 'median_km': median_km, 'max_km': max_km,
            'min_price': min_price, 'median_price': median_price, 'max_price': max_price,
            'price_cv': round(price_cv, 4),
            'price_age_corr': round(price_age_corr, 4) if price_age_corr is not None else None,
            'current_hp_support': hp_support, 'current_hp_source': hp_source, 'current_hp_method': hp_method,
            'source_distribution': source_distribution,
            'arabam_count': arabam_n, 'kaggle_count': kaggle_n,
            'in_ui_catalog': in_catalog,
        })

    return pd.DataFrame(rows)


def _base_priority(train_real_count):
    for label, lo, hi in COUNT_BUCKETS:
        if lo <= train_real_count <= hi:
            return label
    return 'NONE'


def classify_and_recommend(df, premium_threshold, low_price_threshold):
    out_rows = []
    for r in df.to_dict('records'):
        base = _base_priority(r['train_real_count'])
        base_idx = PRIORITY_ORDER.index(base)

        increase_reasons, decrease_reasons = [], []
        median_price = r['median_price'] or 0.0

        if median_price >= premium_threshold:
            increase_reasons.append(f'premium fiyat segmenti (medyan {median_price:,.0f} >= p{int(PREMIUM_PERCENTILE*100)} esigi {premium_threshold:,.0f})')
        if r['current_hp_support'] < LOW_SUPPORT_THRESHOLD:
            increase_reasons.append(f'hierarchical_price destegi dusuk (n={r["current_hp_support"]})')
        if r['current_hp_source'] != 'brand_model':
            increase_reasons.append(f'hierarchical_price {r["current_hp_source"]} katmanina duesuyor (kendi brand_model verisi yok/yetersiz)')
        if r['unique_year_count'] <= FEW_YEARS_THRESHOLD:
            increase_reasons.append(f'model yili cesitliligi cok dusuk ({r["unique_year_count"]} farkli yil)')
        if r['min_price'] and r['max_price'] and r['min_price'] > 0 and (r['max_price'] / r['min_price']) >= WIDE_PRICE_RATIO:
            increase_reasons.append(f'fiyat araligi cok genis (max/min={r["max_price"]/r["min_price"]:.1f}x)')
        if r['in_ui_catalog']:
            increase_reasons.append('UI kataloginda secilebilir (kullanici gercekten bu araci secebilir)')

        if r['max_year'] is not None and r['max_year'] < (CURRENT_YEAR - OLD_STALE_MAX_AGE) and r['raw_real_count'] <= 5:
            decrease_reasons.append(f'cok eski model (max_year={r["max_year"]:.0f}), piyasada nadir - dusuk oncelik')
        if median_price > 0 and median_price <= low_price_threshold:
            decrease_reasons.append(f'dusuk fiyat etkisi (medyan {median_price:,.0f} <= p{int(LOW_PRICE_PERCENTILE*100)} esigi {low_price_threshold:,.0f})')
        if r['arabam_count'] >= FRESH_ARABAM_SUFFICIENT:
            decrease_reasons.append(f'zaten yeterli GUNCEL gercek ornek (arabam={r["arabam_count"]})')

        # net ayarlama: en fazla +2 / -1 seviye, NONE (>=20) grubu en fazla LOW'a
        # cikabilir (guvenlik tavani - "kor sekilde tum n<20 gruplari aday yapma"
        # kuralinin simetrigi: tersine de kor sekilde CRITICAL'e sicratma).
        delta = min(2, len(increase_reasons)) - min(1, len(decrease_reasons))
        new_idx = base_idx - delta  # index kucuk = daha acil
        if base == 'NONE':
            new_idx = max(new_idx, PRIORITY_ORDER.index('LOW'))
        new_idx = max(0, min(len(PRIORITY_ORDER) - 1, new_idx))
        priority = PRIORITY_ORDER[new_idx]

        synthetic_candidate = priority != 'NONE'

        # --- synthetic_safe ---
        safety_reasons = []
        safe = True
        if r['train_real_count'] < 2:
            safe = False
            safety_reasons.append('n<2: aralik/varyans tanimlanamaz, interpolasyon icin yeterli nokta yok')
        if r['unique_year_count'] <= 1:
            safe = False
            safety_reasons.append('tek model yili: yil ekseninde interpolasyon bolgesi yok')
        age_explains_variance = r['price_age_corr'] is not None and r['price_age_corr'] <= -0.5
        if r['price_cv'] > HIGH_CV_UNSAFE and not age_explains_variance:
            safe = False
            safety_reasons.append(
                f'fiyat varyasyonu asiri yuksek (cv={r["price_cv"]:.2f}>{HIGH_CV_UNSAFE}) ve yasla '
                f'aciklanamiyor (corr={r["price_age_corr"]}) - veri ici tutarsizlik supheli'
            )
        if r['min_price'] is None or r['max_price'] is None:
            safe = False
            safety_reasons.append('fiyat verisi eksik - interpolasyon araligi kurulamiyor')

        # --- onerilen sentetik sayi (train_real_count bazli hedef, guven/eski/ucuz ayarlamali) ---
        n = r['train_real_count']
        if not synthetic_candidate:
            recommended = 0
        elif n <= 2:
            recommended = 6
        elif n <= 5:
            target_total = min(16, max(12, 3 * n))
            recommended = max(0, target_total - n)
        elif n <= 9:
            target_total = int(round(15 + (n - 6) * (20 - 15) / 3))  # n=6->15 .. n=9->20
            recommended = max(0, target_total - n)
        elif n <= 19:
            # sadece adaylik NONE'un ustunde kaldiysa (yani artiran faktorler
            # onceligi gercekten LOW'un uzerine tasidiysa) tamamlama onerilir.
            if priority in ('CRITICAL', 'HIGH', 'MEDIUM'):
                target_total = int(round(20 + (n - 10) * (25 - 20) / 9))  # n=10->20 .. n=19->25
                recommended = max(0, target_total - n)
            else:
                recommended = 0
        else:
            recommended = 0

        # eski/ucuz azaltici faktorler varsa oneriyi asagi cek (agresif sentetik
        # uretimden kacin - "sentetik veri gercek veriyi ezmemeli").
        if decrease_reasons and recommended > 0:
            recommended = max(0, recommended - min(4, recommended // 2))

        if not safe and recommended > 0:
            recommended = recommended // 2
            safety_reasons.append(f'guvenli olmadigi icin onerilen sayi yariya indirildi (asil oneri {recommended*2} idi)')

        reason_parts = []
        reason_parts.append(f'{r["train_real_count"]} gercek egitim kaydi (ham: {r["raw_real_count"]})')
        if increase_reasons:
            reason_parts.append('; '.join(increase_reasons))
        if decrease_reasons:
            reason_parts.append('azaltici: ' + '; '.join(decrease_reasons))
        if age_explains_variance and r['price_cv'] > HIGH_CV_UNSAFE:
            reason_parts.append(
                f'fiyat yasla guclu iliskili (corr={r["price_age_corr"]}) - genis fiyat araligi normal '
                f'amortisman, veri tutarsizligi DEGIL; sentetik uretim DUZ interpolasyon yerine yil-fiyat '
                f'egrisine gore yapilmali'
            )
        if not synthetic_candidate:
            reason_parts = [f'{r["train_real_count"]} gercek kayit var; sentetik veri gerekli degil.']
        reason = '. '.join(reason_parts) + '.'

        interpolation_year_range = f'{r["min_year"]:.0f}-{r["max_year"]:.0f}' if r['min_year'] is not None else 'yok'
        interpolation_km_range = f'{r["min_km"]:,.0f}-{r["max_km"]:,.0f}' if r['min_km'] is not None else 'yok'
        interpolation_price_range = f'{r["min_price"]:,.0f}-{r["max_price"]:,.0f}' if r['min_price'] is not None else 'yok'

        out = dict(r)
        out.update({
            'synthetic_candidate': synthetic_candidate,
            'priority': priority,
            'recommended_synthetic_count': int(recommended),
            'reason': reason,
            'synthetic_safe': safe,
            'safety_reason': '; '.join(safety_reasons) if safety_reasons else '',
            'interpolation_year_range': interpolation_year_range,
            'interpolation_km_range': interpolation_km_range,
            'interpolation_price_range': interpolation_price_range,
        })
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def write_reports(df):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    full_path = os.path.join(REPORTS_DIR, 'synthetic_candidates_full.csv')
    priority_path = os.path.join(REPORTS_DIR, 'synthetic_candidates_priority.csv')
    summary_path = os.path.join(REPORTS_DIR, 'synthetic_candidates_summary.md')

    df_sorted = df.sort_values(
        by=['priority', 'train_real_count'],
        key=lambda col: col.map({p: i for i, p in enumerate(PRIORITY_ORDER)}) if col.name == 'priority' else col,
    )
    df_sorted.to_csv(full_path, index=False, encoding='utf-8-sig')

    priority_df = df_sorted[df_sorted['synthetic_candidate']].copy()
    priority_df.to_csv(priority_path, index=False, encoding='utf-8-sig')

    def fmt_row(r):
        safe_txt = 'evet' if r['synthetic_safe'] else 'HAYIR'
        return (f"| {r['priority']} | {r['marka']} | {r['model']} | {r['train_real_count']} | "
                f"{r['unique_year_count']} | {r['median_price']:,.0f} | {r['current_hp_support']} "
                f"({r['current_hp_source']}) | {r['recommended_synthetic_count']} | {safe_txt} | {r['reason']} |")

    top30 = df_sorted[df_sorted['priority'].isin(['CRITICAL', 'HIGH'])].head(30)
    if len(top30) < 30:
        top30 = pd.concat([top30, df_sorted[~df_sorted.index.isin(top30.index)].head(30 - len(top30))])

    premium_table = df_sorted[
        (df_sorted['synthetic_candidate']) &
        (df_sorted['current_hp_support'] < LOW_SUPPORT_THRESHOLD) &
        (df_sorted['median_price'] >= df_sorted['median_price'].quantile(PREMIUM_PERCENTILE))
    ].head(40)

    strong_groups = df_sorted[df_sorted['train_real_count'] >= 20].sort_values('train_real_count', ascending=False).head(8)

    risky_groups = df_sorted[~df_sorted['synthetic_safe']].sort_values('train_real_count').head(40)

    lines = []
    lines.append('# Sentetik Veri Aday Analizi (Faz 30)\n')
    lines.append(f'Toplam brand_model grubu: **{len(df)}**\n')
    lines.append(f'Premium fiyat esigi (p{int(PREMIUM_PERCENTILE*100)}, veri-turevli): '
                 f'{df["median_price"].quantile(PREMIUM_PERCENTILE):,.0f} TL medyan fiyat\n')
    lines.append(f'Dusuk-fiyat esigi (p{int(LOW_PRICE_PERCENTILE*100)}, veri-turevli): '
                 f'{df["median_price"].quantile(LOW_PRICE_PERCENTILE):,.0f} TL medyan fiyat\n')

    lines.append('\n## TABLO 1 — En kritik sentetik veri adaylari (CRITICAL/HIGH once)\n')
    lines.append('| Priority | Marka | Model | Gercek ilan | Yil cesitliligi | Medyan fiyat | HP support | Onerilen sentetik | Safe? | Neden |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for _, r in top30.iterrows():
        lines.append(fmt_row(r))

    lines.append('\n## TABLO 2 — Premium/luxury dusuk destekli modeller (veri-turevli, marka hardcode YOK)\n')
    lines.append('| Priority | Marka | Model | Gercek ilan | Yil cesitliligi | Medyan fiyat | HP support | Onerilen sentetik | Safe? | Neden |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for _, r in premium_table.iterrows():
        lines.append(fmt_row(r))
    if len(premium_table) == 0:
        lines.append('| - | - | - | - | - | - | - | - | - | (esikleri karsilayan grup bulunamadi) |')

    lines.append('\n## TABLO 3 — Sentetik veri GEREKMEYEN guclu gruplar (kontrol amacli, n>=20)\n')
    lines.append('| Marka | Model | Gercek ilan (train) | Medyan fiyat | HP support | Priority |')
    lines.append('|---|---|---|---|---|---|')
    for _, r in strong_groups.iterrows():
        lines.append(f"| {r['marka']} | {r['model']} | {r['train_real_count']} | {r['median_price']:,.0f} | "
                      f"{r['current_hp_support']} | {r['priority']} |")

    lines.append('\n## TABLO 4 — Riskli / sentetik uretmeye UYGUN OLMAYAN gruplar (synthetic_safe=false)\n')
    lines.append('| Marka | Model | Gercek ilan | Yil cesitliligi | price_cv | Neden guvensiz |')
    lines.append('|---|---|---|---|---|---|')
    for _, r in risky_groups.iterrows():
        lines.append(f"| {r['marka']} | {r['model']} | {r['train_real_count']} | {r['unique_year_count']} | "
                      f"{r['price_cv']:.2f} | {r['safety_reason']} |")

    lines.append('\n## OZEL KONTROL — belirlenen 5 arac\n')
    lines.append('| Marka | Model | raw | train | Yil araligi | Km araligi | Fiyat araligi | HP support | Priority | Onerilen sentetik | Safe? |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for marka, model in SPECIAL_CHECK_MODELS:
        match = df_sorted[(df_sorted['marka'] == marka) & (df_sorted['model'] == model)]
        if len(match) == 0:
            lines.append(f'| {marka} | {model} | - | - | - | - | - | - | - | - | dataset\'te bulunamadi |')
            continue
        r = match.iloc[0]
        safe_txt = 'evet' if r['synthetic_safe'] else 'HAYIR'
        lines.append(f"| {marka} | {model} | {r['raw_real_count']} | {r['train_real_count']} | "
                      f"{r['interpolation_year_range']} | {r['interpolation_km_range']} | "
                      f"{r['interpolation_price_range']} | {r['current_hp_support']} ({r['current_hp_source']}) | "
                      f"{r['priority']} | {r['recommended_synthetic_count']} | {safe_txt} |")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return full_path, priority_path, summary_path


def main():
    print('Ham veri okunuyor:', TRAIN_PATH)
    raw_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f'  {len(raw_df)} ham satir')

    print('Production preprocessing uygulaniyor (preprocess.load_clean_train_dataset - SADECE OKUMA)...')
    clean_df = load_clean_train_dataset()
    print(f'  {len(clean_df)} temizlenmis (egitime giren) satir')

    print('Production hierarchical_price artefakti okunuyor (SADECE OKUMA):', MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    hp_lookup = artifact['hierarchical_price']

    catalog_models = _load_catalog_models()
    print(f'  UI katalogunda {len(catalog_models)} marka+model kombinasyonu bulundu')

    print('Marka+model gruplari hesaplaniyor...')
    df = build_group_table(raw_df, clean_df, hp_lookup, catalog_models)

    premium_threshold = df['median_price'].quantile(PREMIUM_PERCENTILE)
    low_price_threshold = df['median_price'].quantile(LOW_PRICE_PERCENTILE)
    print(f'  premium esigi (p{int(PREMIUM_PERCENTILE*100)}): {premium_threshold:,.0f} TL')
    print(f'  dusuk-fiyat esigi (p{int(LOW_PRICE_PERCENTILE*100)}): {low_price_threshold:,.0f} TL')

    df = classify_and_recommend(df, premium_threshold, low_price_threshold)

    full_path, priority_path, summary_path = write_reports(df)

    total = len(df)
    candidates = df[df['synthetic_candidate']]
    counts = df['priority'].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    safe_true = int(df['synthetic_safe'].sum())
    safe_false = int((~df['synthetic_safe']).sum())
    total_recommended = int(df['recommended_synthetic_count'].sum())

    print('\n=== OZET ===')
    print(f'Toplam brand_model grubu: {total}')
    print(f'Sentetik aday: {len(candidates)}')
    for p in PRIORITY_ORDER:
        print(f'{p}: {counts[p]}')
    print(f'Synthetic safe=true: {safe_true}')
    print(f'Synthetic safe=false: {safe_false}')
    print(f'Onerilen toplam sentetik kayit: {total_recommended}')
    print(f'\nRaporlar yazildi:\n  {full_path}\n  {priority_path}\n  {summary_path}')


if __name__ == '__main__':
    main()
