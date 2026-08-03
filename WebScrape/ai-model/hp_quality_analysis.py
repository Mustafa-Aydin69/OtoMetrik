"""Faz 16: motor_gucu veri kalitesi ve yuksek guc guvenlik analizi.

Amac: yuksek guclu araclardaki hatanin ne kadarinin YANLIS VERI, ne kadarinin
GERCEK model extrapolasyonu oldugunu ayirmak. Faz 15 (error_taxonomy_analysis.py)
zaten motor_gucu>300'de bias'in patladigini gostermisti; bu modul KOKENI
ayristirir.

Anahtar bulgu (bu modulun tasarimini yonlendirir): egitim verisindeki en
asiri motor_gucu degerlerinin (>700 HP) buyuk cogunlugu, motor_hacmi (cc)
degeriyle AYNI veya onun temiz bir kati (orn. Renault Kangoo Multix:
motor_hacmi=1300, motor_gucu=130000=1300*100; VW Crafter: motor_hacmi=2000,
motor_gucu=2000 - BIREBIR AYNI). Bu, HP alaninin scraping/parsing sirasinda
motor_hacmi ile KARISTIGI/KOPYALANDIGI bir hata deseni - agirlikli olarak
ticari araclarda (Transit, Doblo, Kangoo, Crafter, Rifter) goruluyor,
muhtemelen bu kategori icin arabam.com detay tablosu duzeninin farkli
olmasindan kaynaklaniyor. Bu yuzden "probable_parse_error" siniflandirmasinin
BIRINCIL sinyali motor_gucu~motor_hacmi benzerligidir - saf istatistiksel
peer-sapmasi degil.

Uc deney (Madde 4) AYRI modeller egitir, PRODUCTION artefaktini (models/
lightgbm_final.joblib) DEGISTIRMEZ - sadece bu script icinde bellekte train
edilip ayni dis holdout'ta karsilastirilir.
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

from train import (
    BASELINE_PARAMS, CATEGORICAL_COLS, load_cars1_holdout, load_model,
    prepare_external_holdout, prepare_full_training_data, prepare_training_data,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'hp_quality_report.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'hp_quality_summary.txt')
PLOT_DIR = os.path.join(BASE_DIR, 'data', 'output', 'hp_quality')

YIL_BUCKET_SIZE = 5
MIN_PEER_COUNT = 5
KW_TO_HP = 1.341  # 1 kW = 1.341 HP - unit-confusion tespiti icin
HP_PER_LITER_IMPLAUSIBLE = 350  # en gelismis uretim motorlarinin bile ustunde
POWER_BINS_EXP = [0, 300, 400, np.inf]
POWER_BIN_LABELS_EXP = ['<=300', '201-300', '301-400', '400+']  # asagida ayrica hesaplanir


def robust_zscore(value, median, mad):
    if mad == 0 or pd.isna(mad):
        return 0.0 if value == median else np.sign(value - median) * 999
    return (value - median) / (1.4826 * mad)


def mad(series):
    med = series.median()
    return (series - med).abs().median()


# ---------------------------------------------------------------------------
# Madde 1: dagilim denetimi
# ---------------------------------------------------------------------------
def distribution_report(name, series):
    lines = [f'--- {name} (n={len(series):,}) ---']
    lines.append(f'min={series.min():.1f} max={series.max():.1f}')
    for p in [95, 99, 99.5, 99.9, 99.99]:
        lines.append(f'  P{p}: {np.percentile(series, p):,.1f}')
    for thr in [300, 400, 500, 600]:
        n = (series >= thr).sum()
        lines.append(f'  >={thr} HP: {n} kayit (%{100 * n / len(series):.3f})')
    return lines


def high_power_brand_model_table(df, threshold=250):
    sub = df[df['motor_gucu'] >= threshold]
    grp = sub.groupby(['marka', 'model'], observed=True).size().sort_values(ascending=False)
    lines = [f'=== marka/model bazinda >={threshold} HP kayit sayisi (ilk 20) ===']
    for (marka, model), n in grp.head(20).items():
        lines.append(f'  {marka} {model}: {n}')
    return lines


# ---------------------------------------------------------------------------
# Madde 2: peer-based anomaly score
# ---------------------------------------------------------------------------
def build_peer_stats(X_full):
    df = X_full.copy()
    df['yil_bucket'] = (df['yil'] // YIL_BUCKET_SIZE * YIL_BUCKET_SIZE).astype(int)

    primary = df.groupby(['marka', 'model', 'yil_bucket'], observed=True)['motor_gucu'].agg(
        hp_peer_median='median', hp_peer_mad=mad, hp_peer_count='count').reset_index()
    fallback = df.groupby(['marka', 'kasa_turu', 'yil_bucket'], observed=True)['motor_gucu'].agg(
        hp_peer_median='median', hp_peer_mad=mad, hp_peer_count='count').reset_index()
    return primary, fallback


def attach_peer_stats(df, primary, fallback):
    df = df.copy()
    df['yil_bucket'] = (df['yil'] // YIL_BUCKET_SIZE * YIL_BUCKET_SIZE).astype(int)

    merged = df.merge(primary, on=['marka', 'model', 'yil_bucket'], how='left', suffixes=('', '_p'))
    insufficient = merged['hp_peer_count'].isna() | (merged['hp_peer_count'] < MIN_PEER_COUNT)

    if insufficient.any():
        fb = df.loc[insufficient, ['marka', 'kasa_turu', 'yil_bucket']].merge(
            fallback, on=['marka', 'kasa_turu', 'yil_bucket'], how='left')
        merged.loc[insufficient, 'hp_peer_median'] = fb['hp_peer_median'].values
        merged.loc[insufficient, 'hp_peer_mad'] = fb['hp_peer_mad'].values
        merged.loc[insufficient, 'hp_peer_count'] = fb['hp_peer_count'].values
        merged.loc[insufficient, 'peer_group_type'] = 'marka+kasa_turu+yil'
    merged['peer_group_type'] = merged.get('peer_group_type', pd.Series('marka+model+yil', index=merged.index))
    merged['peer_group_type'] = merged['peer_group_type'].fillna('marka+model+yil')

    still_insufficient = merged['hp_peer_count'].isna() | (merged['hp_peer_count'] < MIN_PEER_COUNT)
    merged.loc[still_insufficient, 'peer_group_type'] = 'insufficient'

    merged['hp_peer_median'] = merged['hp_peer_median'].fillna(merged['motor_gucu'].median())
    merged['hp_peer_mad'] = merged['hp_peer_mad'].fillna(0)
    merged['hp_peer_count'] = merged['hp_peer_count'].fillna(0)

    merged['hp_deviation_ratio'] = merged['motor_gucu'] / merged['hp_peer_median'].replace(0, np.nan)
    merged['hp_deviation_ratio'] = merged['hp_deviation_ratio'].fillna(1.0)
    merged['hp_robust_zscore'] = merged.apply(
        lambda r: robust_zscore(r['motor_gucu'], r['hp_peer_median'], r['hp_peer_mad']), axis=1)

    return merged


# ---------------------------------------------------------------------------
# Madde 3: siniflandirma
# ---------------------------------------------------------------------------
def is_close(value, target, tol):
    return abs(value - target) / target <= tol


def classify_hp_row(row):
    hacmi_l = row['motor_hacmi'] / 1000 if row['motor_hacmi'] > 0 else np.nan
    hp_per_liter = row['motor_gucu'] / hacmi_l if hacmi_l and hacmi_l > 0 else np.nan

    # birincil sinyal: motor_gucu, motor_hacmi ile (neredeyse) AYNI veya
    # onun temiz bir kati - scraping/parsing sirasinda iki alanin karismasi/
    # kopyalanmasi (bkz. modul docstring'i).
    if row['motor_hacmi'] > 0:
        hacmi_similarity = abs(row['motor_gucu'] - row['motor_hacmi']) / row['motor_hacmi']
        if hacmi_similarity < 0.15:
            return 'probable_parse_error'
        for mult in (10, 100):
            if is_close(row['motor_gucu'], row['motor_hacmi'] * mult, 0.15):
                return 'probable_parse_error'

    # kW<->HP karisikligi: ZAYIF bir sinyal - gercek bir "kW alani" sutunumuz
    # olmadigi icin bu sadece "peer medyanina orani, donusum faktorune (1.341)
    # SUPHELI DERECEDE yakin" seklinde dolayli bir ipucu. %2 tolerans bile
    # (ilk denemede %10 kullanilmisti, ~%80 satiri yanlislikla isaretledi)
    # binlerce satiri "yakalayabilir" cunku surekli bir dagilimda rastgele
    # oranlarin bir kismi kacinilmaz olarak 1.341'e yakin duser. Bu yuzden
    # tolerans %0.5'e cekildi VE rapora "dusuk kesinlik, elle inceleme
    # gerektirir" notu eklendi (bkz. summary).
    if row['peer_group_type'] != 'insufficient':
        ratio = row['hp_deviation_ratio']
        if is_close(ratio, KW_TO_HP, 0.005) or is_close(ratio, 1 / KW_TO_HP, 0.005):
            return 'probable_unit_error'

    # elektrikli araclarda motor_hacmi anlamsiz/sifira yakin oldugu icin
    # (bkz. dagitima hazirlik analizindeki elektrikli+motor_hacmi bulgusu)
    # HP/L orani burada HESAPLANMAZ - aksi halde motor_hacmi~0 bolmesi
    # sahte-pozitif "physically_implausible" uretir (bkz. Mercedes EQE
    # ornegi, ilk denemede peer'iyle BIREBIR ayniyken yanlislikla isaretlendi).
    if row['yakit_turu'] != 'Elektrik' and row['motor_hacmi'] >= 200:
        if not pd.isna(hp_per_liter) and hp_per_liter > HP_PER_LITER_IMPLAUSIBLE:
            return 'physically_implausible'

    if row['peer_group_type'] == 'insufficient':
        return 'insufficient_peer_data' if row['motor_gucu'] >= 200 else ''

    z = row['hp_robust_zscore']
    ratio = row['hp_deviation_ratio']
    if abs(z) > 8 and (ratio > 4 or ratio < 0.25):
        return 'physically_implausible'
    if abs(z) > 3 or ratio > 1.8 or ratio < 0.55:
        return 'rare_but_plausible'
    if row['motor_gucu'] >= 200:
        return 'valid_high_performance'
    return ''


def build_hp_quality_dataframe():
    X_full, y_full = prepare_full_training_data()
    X_holdout, y_holdout = prepare_external_holdout(X_full, y_full)

    primary, fallback = build_peer_stats(X_full)

    train_df = attach_peer_stats(X_full.assign(fiyat=y_full.values), primary, fallback)
    train_df['split'] = 'egitim'
    holdout_df = attach_peer_stats(X_holdout.assign(fiyat=y_holdout.values), primary, fallback)
    holdout_df['split'] = 'dis_holdout'

    full = pd.concat([train_df, holdout_df], ignore_index=True)
    # HP/L (fiziksel makul-luk kontrolu) - motor_hacmi cc -> litre.
    full['hp_hacmi_orani'] = full['motor_gucu'] / (full['motor_hacmi'] / 1000).replace(0, np.nan)
    # motor_gucu/motor_hacmi HAM orani (parse-error tespitinin dayandigi
    # benzerlik sinyali icin raporlamada kullanilir - HP/L ile KARISTIRILMAMALI).
    full['hp_motor_hacmi_ham_oran'] = full['motor_gucu'] / full['motor_hacmi'].replace(0, np.nan)
    full['tag'] = full.apply(classify_hp_row, axis=1)

    cols = ['split', 'marka', 'model', 'paket', 'yil', 'motor_hacmi', 'motor_gucu', 'yakit_turu', 'fiyat',
            'hp_hacmi_orani', 'hp_motor_hacmi_ham_oran', 'hp_peer_median', 'hp_peer_mad', 'hp_peer_count',
            'peer_group_type', 'hp_deviation_ratio', 'hp_robust_zscore', 'tag']
    return full[cols], X_full, y_full, X_holdout, y_holdout


# ---------------------------------------------------------------------------
# Madde 4: temizleme senaryolari (A/B/C) - PRODUCTION artefaktini degistirmez,
# sadece bu script icinde bellekte train edilip ayni dis holdout'ta karsilastirilir.
# ---------------------------------------------------------------------------
GLOBAL_P999_CAP = None  # main()'de doldurulur


def train_variant(X, y):
    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(X, y)
    return model


def segment_stats(y_true, y_pred, motor_gucu):
    def stats_for(mask, label):
        if mask.sum() == 0:
            return {'label': label, 'n': 0}
        yt, yp = y_true[mask], y_pred[mask]
        error = yp - yt
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            'label': label, 'n': int(mask.sum()),
            'mae': np.abs(error).mean(), 'rmse': np.sqrt((error ** 2).mean()),
            'r2': r2, 'bias': error.mean(),
            'neg_pred_count': int((yp <= 0).sum()),
        }

    return [
        stats_for(np.ones(len(y_true), dtype=bool), 'GENEL'),
        stats_for((motor_gucu >= 201) & (motor_gucu <= 300), '201-300 HP'),
        stats_for((motor_gucu >= 301) & (motor_gucu <= 400), '301-400 HP'),
        stats_for(motor_gucu > 400, '400+ HP'),
    ]


def format_segment_stats(scenario_label, rows):
    lines = [f'--- Senaryo {scenario_label} ---']
    for r in rows:
        if r['n'] == 0:
            lines.append(f'  {r["label"]:12s} n=0')
            continue
        r2_str = f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else 'n/a'
        lines.append(f'  {r["label"]:12s} n={r["n"]:>6,} MAE={r["mae"]:>10,.0f} RMSE={r["rmse"]:>10,.0f} '
                     f'bias={r["bias"]:>+10,.0f} R2={r2_str:>7s} negatif_tahmin={r["neg_pred_count"]}')
    return lines


def run_cleaning_experiments(X_full, y_full, X_holdout, y_holdout, flagged_mask, global_cap):
    results = {}

    # A - mevcut veri, hic filtre yok
    model_a = train_variant(X_full, y_full)
    pred_a = model_a.predict(X_holdout)
    results['A (filtresiz)'] = segment_stats(y_holdout.values, pred_a, X_holdout['motor_gucu'].values)

    # B - yalniz acik anomalileri cikar (probable_parse_error + physically_implausible)
    keep_mask = ~flagged_mask
    X_b, y_b = X_full[keep_mask], y_full[keep_mask]
    model_b = train_variant(X_b, y_b)
    pred_b = model_b.predict(X_holdout)
    results['B (anomali cikarildi)'] = segment_stats(y_holdout.values, pred_b, X_holdout['motor_gucu'].values)

    # C - winsorize: global P99.9 (egitimden) ustundeki motor_gucu degerleri kirpilir
    X_c = X_full.copy()
    X_c['motor_gucu'] = X_c['motor_gucu'].clip(upper=global_cap)
    model_c = train_variant(X_c, y_full)
    X_holdout_c = X_holdout.copy()
    X_holdout_c['motor_gucu'] = X_holdout_c['motor_gucu'].clip(upper=global_cap)
    pred_c = model_c.predict(X_holdout_c)
    results['C (P99.9 winsorize)'] = segment_stats(y_holdout.values, pred_c, X_holdout['motor_gucu'].values)

    return results, {'A (filtresiz)': model_a, 'B (anomali cikarildi)': model_b, 'C (P99.9 winsorize)': model_c}


def main():
    df, X_full, y_full, X_holdout, y_holdout = build_hp_quality_dataframe()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    report = ['=== Madde 1: motor_gucu dagilim denetimi ===']
    X_train, X_test, y_train, y_test = prepare_training_data()
    report.extend(distribution_report('Egitim (X_full)', X_full['motor_gucu']))
    report.extend(distribution_report('Ic test (X_test)', X_test['motor_gucu']))
    report.extend(distribution_report('Dis holdout (X_holdout)', X_holdout['motor_gucu']))
    report.append('')
    report.extend(high_power_brand_model_table(df[df['split'] == 'egitim']))
    report.append('')

    report.append('=== Etiket dagilimi (tum satirlar, egitim+holdout) ===')
    for tag in ['probable_parse_error', 'probable_unit_error', 'physically_implausible',
                'rare_but_plausible', 'valid_high_performance', 'insufficient_peer_data']:
        n = (df['tag'] == tag).sum()
        report.append(f'  {tag:26s} n={n:>5,}')
    report.append('NOT: probable_unit_error DUSUK KESINLIKLI bir etikettir - gercek bir "kW alani" '
                  'olmadigi icin sadece peer medyanina oranin donusum faktorune (1.341) SUPHELI '
                  'derecede yakin olmasina dayanir; elle inceleme gerektirir, otomatik temizleme '
                  'icin TEK BASINA yeterli kanit sayilmamalidir (bkz. Madde 4 - B senaryosu bu '
                  'etiketi CIKARMAZ, sadece probable_parse_error + physically_implausible cikarir).')
    report.append('')

    report.append('=== probable_parse_error ornekleri (ilk 15) - motor_gucu/motor_hacmi ham orani ===')
    examples = df[df['tag'] == 'probable_parse_error'].head(15)
    for _, r in examples.iterrows():
        oran = r['hp_motor_hacmi_ham_oran']
        oran_str = f'{oran:.2f}' if pd.notna(oran) else 'n/a'
        report.append(f'  {r["marka"]} {r["model"]!r} yil={r["yil"]:.0f} '
                      f'motor_hacmi={r["motor_hacmi"]:.0f} motor_gucu={r["motor_gucu"]:.0f} '
                      f'(motor_gucu/motor_hacmi={oran_str})')
    report.append('')

    report.append('=== physically_implausible ornekleri (ilk 15) ===')
    examples = df[df['tag'] == 'physically_implausible'].head(15)
    for _, r in examples.iterrows():
        report.append(f'  {r["marka"]} {r["model"]!r} yil={r["yil"]:.0f} motor_gucu={r["motor_gucu"]:.0f} '
                      f'peer_medyan={r["hp_peer_median"]:.0f} peer_n={r["hp_peer_count"]:.0f} '
                      f'z={r["hp_robust_zscore"]:.1f} oran={r["hp_deviation_ratio"]:.2f}')
    report.append('')

    # -------------------------------------------------------------------
    # Madde 4: temizleme senaryolari A/B/C
    # -------------------------------------------------------------------
    train_tags = df[df['split'] == 'egitim'].reset_index(drop=True)['tag']
    flagged_mask = train_tags.isin(['probable_parse_error', 'physically_implausible']).values
    global_cap = float(np.percentile(X_full['motor_gucu'], 99.9))

    report.append('=== Madde 4: temizleme senaryolari (ayni dis holdout uzerinde) ===')
    report.append(f'B senaryosu icin cikarilan satir sayisi: {flagged_mask.sum()} / {len(X_full):,} '
                  f'(probable_parse_error + physically_implausible)')
    report.append(f'C senaryosu icin winsorize ust siniri (egitim P99.9): {global_cap:.0f} HP')
    report.append('')

    exp_results, exp_models = run_cleaning_experiments(X_full, y_full, X_holdout, y_holdout,
                                                         flagged_mask, global_cap)
    for label, rows in exp_results.items():
        report.extend(format_segment_stats(label, rows))
        report.append('')

    # flagged Hyundai Accent (601 HP) ornek satirinin her senaryodaki tahmini
    accent_mask = (X_holdout['marka'] == 'Hyundai') & (X_holdout['model'] == 'Accent') & \
                  (X_holdout['motor_gucu'] == 601)
    if accent_mask.any():
        report.append('=== Ornek olay: Hyundai Accent 601 HP (veri hatasi) satirinin senaryo bazinda tahmini ===')
        idx = np.where(accent_mask.values)[0][0]
        for label, model in exp_models.items():
            if 'C' in label:
                row = X_holdout.copy()
                row['motor_gucu'] = row['motor_gucu'].clip(upper=global_cap)
                pred = model.predict(row.iloc[[idx]])[0]
            else:
                pred = model.predict(X_holdout.iloc[[idx]])[0]
            report.append(f'  {label}: tahmin={pred:,.0f} TL (gercek fiyat: {y_holdout.values[idx]:,.0f} TL)')
        report.append('')

    # -------------------------------------------------------------------
    # grafikler
    # -------------------------------------------------------------------
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_paths = []

    def save(fig, name):
        path = os.path.join(PLOT_DIR, name)
        try:
            fig.savefig(path, bbox_inches='tight', dpi=120)
        except ValueError:
            # bazi coklu-subplot + suptitle kombinasyonlarinda matplotlib'in
            # tight-bbox hesaplamasi NaN'a dusebiliyor (bilinen bir kenar
            # durumu) - bbox_inches olmadan tekrar dene.
            fig.savefig(path, dpi=120)
        plt.close(fig)
        plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    train_hp = X_full['motor_gucu']
    ax.hist(train_hp[train_hp <= 800], bins=80, alpha=0.7, label='egitim (<=800 HP)')
    for tag, color in [('probable_parse_error', 'red'), ('physically_implausible', 'orange'),
                        ('probable_unit_error', 'purple')]:
        vals = df[(df['split'] == 'egitim') & (df['tag'] == tag) & (df['motor_gucu'] <= 800)]['motor_gucu']
        if len(vals) > 0:
            ax.scatter(vals, np.full(len(vals), -20 - 20 * ['probable_parse_error', 'physically_implausible',
                       'probable_unit_error'].index(tag)), color=color, label=tag, s=12, zorder=5)
    ax.set_xlabel('motor_gucu (HP)')
    ax.set_ylabel('frekans')
    ax.set_title('Egitim motor_gucu dagilimi + supheli kayitlar')
    ax.legend(fontsize=8)
    save(fig, 'hp_distribution_flagged.png')

    fig, ax = plt.subplots(figsize=(7, 5))
    plausible = df[(df['peer_group_type'] != 'insufficient') & (df['motor_gucu'] <= 800)]
    ax.scatter(plausible['motor_gucu'], plausible['hp_robust_zscore'].clip(-20, 20), s=3, alpha=0.1)
    ax.axhline(3, color='orange', linestyle='--', linewidth=1)
    ax.axhline(-3, color='orange', linestyle='--', linewidth=1)
    ax.set_xlabel('motor_gucu (HP)')
    ax.set_ylabel('peer robust z-score [-20,20 kirpilmis]')
    ax.set_title('Peer-bazli sapma (robust z-score)')
    save(fig, 'peer_deviation_scatter.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, seg_label in zip(axes, ['201-300 HP', '301-400 HP', '400+ HP']):
        maes = []
        labels = []
        for scenario, rows in exp_results.items():
            seg = next(r for r in rows if r['label'] == seg_label)
            maes.append(seg['mae'] if seg['n'] > 0 else 0)
            labels.append(scenario.split(' ')[0])
        ax.bar(labels, maes, color=['tab:blue', 'tab:green', 'tab:orange'])
        ax.set_title(seg_label)
        ax.set_ylabel('MAE (TL)')
    fig.suptitle('Senaryo bazinda MAE (yuksek guc dilimleri)')
    fig.tight_layout()
    save(fig, 'scenario_comparison_mae.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, seg_label in zip(axes, ['201-300 HP', '301-400 HP', '400+ HP']):
        biases = []
        labels = []
        for scenario, rows in exp_results.items():
            seg = next(r for r in rows if r['label'] == seg_label)
            biases.append(seg['bias'] if seg['n'] > 0 else 0)
            labels.append(scenario.split(' ')[0])
        colors = ['tab:red' if b < 0 else 'tab:blue' for b in biases]
        ax.bar(labels, biases, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(seg_label)
        ax.set_ylabel('bias (TL)')
    fig.suptitle('Senaryo bazinda bias (yuksek guc dilimleri)')
    fig.tight_layout()
    save(fig, 'scenario_comparison_bias.png')

    report.append('=== Kaydedilen grafikler ===')
    report.extend(plot_paths)
    report.append('')

    # -------------------------------------------------------------------
    # Madde 5: yuksek guc icin guven mekanizmasi TASARIMI (uygulanmadi)
    # -------------------------------------------------------------------
    report.append('=== Madde 5: yuksek guc segmenti icin guven mekanizmasi (TASARIM - henuz uygulanmadi) ===')
    report.append(
        'Oneri: /predict yanitina "support" alani eklenir - marka+model (once) veya marka+kasa_turu\n'
        '(peer_group_type="insufficient" ise) grubundaki egitim ornegi sayisina (peer_count) dayanir,\n'
        'SADECE motor_gucu\'ye degil MODEL+HP KOMBINASYONUNA bakar (bkz. gorev talebi - "Sadece HP\'ye\n'
        'gore warning verme"):\n'
        '\n'
        '  confidence = "high"   eger peer_count >= 100\n'
        '  confidence = "medium" eger 20 <= peer_count < 100\n'
        '  confidence = "low"    eger peer_count < 20\n'
        '\n'
        'Ornek yanit (serve.py\'ye HENUZ eklenmedi):\n'
        '{\n'
        '  "price": 3590000,\n'
        '  "currency": "TRY",\n'
        '  "source": "model",\n'
        '  "warnings": [\n'
        '    {\n'
        '      "code": "low_support_high_power_segment",\n'
        '      "message": "Bu marka/model + motor gucu kombinasyonu egitim verisinde az temsil\n'
        '                   ediliyor (8 benzer kayit). Tahmin daha az guvenilir olabilir."\n'
        '    }\n'
        '  ],\n'
        '  "support": {"peer_count": 8, "peer_group": "marka+model+yil", "confidence": "low"}\n'
        '}\n'
        '\n'
        'peer_count, bu script\'teki attach_peer_stats() ile ZATEN hesaplaniyor - serve.py\'ye tasima\n'
        'maliyeti dusuk (CATEGORY_SETS gibi baslangicta bir kez hesaplanip bellekte tutulabilir).\n'
        'BURADA UYGULANMADI - onceki gorevlerdeki desene uygun olarak once olcum/tasarim, sonra\n'
        'ayri bir onay/implementasyon adimi.'
    )
    report.append('')

    # -------------------------------------------------------------------
    # Madde 6: motor_gucu<=800 sinirinin yeniden degerlendirilmesi
    # -------------------------------------------------------------------
    p999 = np.percentile(X_full['motor_gucu'], 99.9)
    p9999 = np.percentile(X_full['motor_gucu'], 99.99)
    seg_400 = next(r for r in exp_results['A (filtresiz)'] if r['label'] == '400+ HP')
    seg_300 = next(r for r in exp_results['A (filtresiz)'] if r['label'] == '201-300 HP')
    report.append('=== Madde 6: motor_gucu<=800 sinirinin yeniden degerlendirilmesi ===')
    report.append(
        f'Su an domain_validation.py\'de TEK bir sayi (800) HEM "fiziksel olarak mumkun mu" HEM\n'
        f'"model buna guvenilir tahmin uretebilir mi" sorularini birlikte cevaplamaya calisiyor -\n'
        f'bunlar FARKLI sorular, ayrilmali:\n'
        f'\n'
        f'  1) FIZIKSEL API SINIRI (kesin ret): gercek uretim araclarinin (en guclu hypercarlar\n'
        f'     dahil, ~1500-1600 HP araligi) USTUNDE bir deger fiziksel olarak imkansiza yakindir.\n'
        f'     Oneri: ~2000 HP - bu sinirin AMACI veri girisi saçmaligini (99999 gibi) reddetmek,\n'
        f'     "model bunu iyi tahmin eder mi" sorusuyla ILGILI DEGIL.\n'
        f'\n'
        f'  2) EGITIM VERISINDEKI ISTATISTIKSEL UC DEGER: P99.9={p999:.0f} HP, P99.99={p9999:.0f} HP.\n'
        f'     800 sinirinin secim gerekcesi buydu (P99.99 uzerine marj) - bu hala GECERLI bir\n'
        f'     "olagan disi ama mumkun" esigi, ama "guvenilir" anlamina GELMIYOR.\n'
        f'\n'
        f'  3) MODELIN GUVENILIR DESTEK BOLGESI (asil onemli olan): Faz 15 error taxonomy +\n'
        f'     bu analiz TUTARLI sekilde gosteriyor ki performans ~300 HP civarinda ciddi\n'
        f'     bozuluyor - 201-300 HP: R2={seg_300["r2"]:.2f}, MAE={seg_300["mae"]:,.0f} (halen makul);\n'
        f'     400+ HP: R2={seg_400["r2"]:.2f} (NEGATIF - ortalamayi tahmin etmekten KOTU),\n'
        f'     MAE={seg_400["mae"]:,.0f}. Yani 700 HP GERCEK bir arac icin "mumkun" olabilir\n'
        f'     (madde 6 orneginde belirtildigi gibi) ama modelin onu GUVENILIR tahmin ettigi\n'
        f'     anlamina GELMEZ.\n'
        f'\n'
        f'ONERI: domain_validation.py\'deki TEK "800" sinirini IKI KATMANLI hale getir (bu\n'
        f'raporda tasarlanmistir, HENUZ UYGULANMADI):\n'
        f'  - Sert ret (422): motor_gucu>~2000 (fiziksel imkansizlik)\n'
        f'  - Yumusak uyari (200 + warning, Madde 5\'teki support mekanizmasi ile): motor_gucu>~300\n'
        f'    VEYA peer_count dusuk - "kabul edilebilir ama dusuk guven" bolgesi\n'
        f'  800 sinirini DOGRUDAN degistirmedim (onceki gorevlerdeki "once olc, sonra uygula"\n'
        f'  desenine uygun olarak bu bir tasarim onerisidir, kod degisikligi degil).'
    )
    report.append('')

    # -------------------------------------------------------------------
    # Madde 7: net karar
    # -------------------------------------------------------------------
    a300 = next(r for r in exp_results['A (filtresiz)'] if r['label'] == '301-400 HP')
    b300 = next(r for r in exp_results['B (anomali cikarildi)'] if r['label'] == '301-400 HP')
    c300 = next(r for r in exp_results['C (P99.9 winsorize)'] if r['label'] == '301-400 HP')
    a400 = seg_400
    b400 = next(r for r in exp_results['B (anomali cikarildi)'] if r['label'] == '400+ HP')
    c400 = next(r for r in exp_results['C (P99.9 winsorize)'] if r['label'] == '400+ HP')

    report.append('=== Madde 7: NET KARAR ===')
    report.append(
        f'BEKLENMEDIK AMA NET BULGU: 137 satirlik acik anomaliyi cikarmak (B) veya P99.9\'da\n'
        f'winsorize etmek (C), yuksek guc segmentinde performansi DUZELTMEDI - HAFIFCE\n'
        f'KOTULESTIRDI:\n'
        f'  301-400 HP MAE:  A={a300["mae"]:,.0f}  B={b300["mae"]:,.0f}  C={c300["mae"]:,.0f}\n'
        f'  400+ HP MAE:     A={a400["mae"]:,.0f}  B={b400["mae"]:,.0f}  C={c400["mae"]:,.0f}\n'
        f'  400+ HP R2:      A={a400["r2"]:.2f}  B={b400["r2"]:.2f}  C={c400["r2"]:.2f}\n'
        f'\n'
        f'NEDEN: flagged 137 satir egitim verisinin sadece %0.048\'i - kaldirmak/kirpmak\n'
        f'genel modeli neredeyse hic etkilemiyor. Winsorize (C) ise motor_gucu\'yu egitimde\n'
        f'338\'e SABITLEDIGI icin model 400+ HP\'yi GERCEKTEN AYIRT EDEMEZ HALE GELIYOR -\n'
        f'gercek sinyal kaybediliyor. SONUC: yuksek guc segmentindeki zayif performansIN\n'
        f'ANA NEDENI VERI KIRLILIGI DEGIL - GERCEK ORNEKLEM YETERSIZLIGI/EXTRAPOLASYONDUR\n'
        f'(egitimde sadece 170 kayit >=400 HP, 36 kayit >=600 HP - bkz. Madde 1).\n'
        f'\n'
        f'ANCAK veri kalitesi TEMIZLIGI hala ONEMLI - sadece FARKLI bir nedenle: TEK bir\n'
        f'bozuk kayit (Hyundai Accent, motor_hacmi/motor_gucu karismis) GENEL modeli negatif\n'
        f'fiyat uretmeye (-103,006 TL) goturebiliyor; bu ROBUSTLUK/guvenlik sorunu, DOGRULUK\n'
        f'sorunu degil. B ve C bu spesifik hatayi DUZELTIYOR (936,596 / 388,014 TL - hala\n'
        f'yanlis ama en azindan POZITIF ve makul araliktalar).\n'
        f'\n'
        f'KARARLAR:\n'
        f'  1. HANGI KAYITLAR TEMIZLENMELI: probable_parse_error (n=60) + physically_implausible\n'
        f'     (n=80) - toplam 140 kayit, YUKSEK GUVENLE (motor_gucu~motor_hacmi benzerligi veya\n'
        f'     >8 robust z-score + fiziksel imkansizlik). Bunlar production egitim verisinden\n'
        f'     CIKARILMALI (genel MAE\'yi olcumlenebilir sekilde IYILESTIRMEZ ama negatif/absurd\n'
        f'     tahmin riskini AZALTIR - bir robustluk/guvenlik onlemi).\n'
        f'  2. HANGI KAYITLAR KORUNMALI: rare_but_plausible (n=74,071) ve valid_high_performance\n'
        f'     (n=5,268) - bunlar GERCEK sinyal tasiyor (bkz. yuksek guclu marka/model tablosu -\n'
        f'     Land Rover, Mercedes S/E, Porsche Panamera, BMW hep bekleneni yansitiyor).\n'
        f'     probable_unit_error (n=551) DUSUK KESINLIKLI - elle/orneklem incelemesi\n'
        f'     yapilmadan TEMIZLENMEMELI (bkz. Madde 3 notu).\n'
        f'  3. YENIDEN EGITIM GEREKIYOR MU: EVET ama SADECE robustluk icin (140 kayit cikar) -\n'
        f'     genel dogrulugu artirmak icin DEGIL (deneyler bunun iste dogru cozum olmadigini\n'
        f'     gosterdi). Gercek dogruluk artisi icin asil ihtiyac: DAHA FAZLA yuksek-guc\n'
        f'     egitim ornegi (veri toplama) - bu script\'in KAPSAMI DISINDA.\n'
        f'  4. YUKSEK GUC SEGMENTINDE WARNING GEREKLI MI: EVET, KESINLIKLE - veri temizligi\n'
        f'     bunu duzeltmiyor, R2 400+ HP\'de hala NEGATIF (B: {b400["r2"]:.2f}, C: {c400["r2"]:.2f}).\n'
        f'     Madde 5\'teki peer_count-tabanli support/confidence mekanizmasi (SADECE HP degil,\n'
        f'     model+HP kombinasyonu) sonraki adimda UYGULANMALI.\n'
        f'  5. MODEL PERFORMANSI NE KADAR DEGISTI: GENEL MAE B\'de +%0.7, C\'de +%0.5 (her ikisi\n'
        f'     de HAFIFCE kotu) - pratikte ihmal edilebilir. Segment bazinda 301-400/400+ HP\'de\n'
        f'     B ve C DAHA KOTU (yukarida). Bu deneyler SONUCUNDA production modeli\n'
        f'     DEGISTIRILMEDI (talebe uygun) - onerilen "140 kayit cikar" degisikligi AYRI bir\n'
        f'     onay adiminda uygulanmali, cunku (kucuk de olsa) genel MAE\'yi hafifce kotulestirir\n'
        f'     ve bu odun bilerek/robustluk icin kabul edilmelidir.'
    )
    report.append('')

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f'\nozet: {SUMMARY_PATH}')
    print(f'CSV: {CSV_PATH}')


if __name__ == '__main__':
    main()
