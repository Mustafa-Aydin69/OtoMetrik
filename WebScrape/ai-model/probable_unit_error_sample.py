"""Faz 24: hp_quality_analysis.py'nin 'probable_unit_error' etiketli n=551
kaydi (Madde 3 - DUSUK KESINLIKLI: gercek bir "kW alani" olmadigi icin sadece
peer medyanina oranin 1.341 kW->HP donusum faktorune SUPHELI derecede yakin
olmasina dayanir) icin SKOR-TABAKALI manuel inceleme ornek kumesi uretir.

AMAC 100 kaydi temizlemek DEGIL - 551 kaydin OTOMATIK olarak filtrelenip
filtrelenemeyecegine karar vermek icin yeterli KANIT toplamak (bkz. gorev
talebi). Bu yuzden bu script:
  - OTOMATIK SINIFLANDIRMA YAPMAZ - manual_siniflandirma/manual_not sutunlari
    BOS birakilir, karar TAMAMEN insan incelemesine birakilir.
  - Production modelini veya egitim verisini (train_dataset.csv,
    models/lightgbm_final.joblib) HICBIR SEKILDE DEGISTIRMEZ - sadece MEVCUT
    (zaten diskte olan) production artefaktiyla bu 100 kayit icin TAHMIN
    hesaplar (baglam icin - "gercek fiyat ne kadar sapiyor" sorusuna cevap).

ORNEKLEM TASARIMI (duz rastgele 100 DEGIL - dusuk/yuksek supheli skor
dagilimini kacirmamak icin): 551 kaydin |hp_robust_zscore| dagilimi TERCILE'lara
(high/medium/low) bolunur, HER TABAKADAN rastgele (seed=42) secilir:
  40 en yuksek tabaka (high)  + 30 orta tabaka (medium) + 20 en dusuk tabaka (low)
  + 10 KALAN havuzdan tamamen rastgele kontrol grubu (yukaridaki 90'i DISLAR).
Bu, "sadece en asiri 40/orta 30/en dusuk 20'yi SIRALI al" yaklasimindan
farkli - her tabaka KENDI ICINDE rastgele orneklenir, boylece tabaka ICI
cesitlilik de temsil edilir (sadece siralamanin en/orta/az uc noktalari degil).

Calistirma (ai-model/ calisma dizini olarak): python probable_unit_error_sample.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from hp_quality_analysis import KW_TO_HP, build_hp_quality_dataframe
from train import apply_saved_categories, load_model

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'probable_unit_error_sample.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'probable_unit_error_sample_summary.txt')

SAMPLE_SEED = 42
N_HIGH, N_MEDIUM, N_LOW, N_RANDOM = 40, 30, 20, 10
MANUAL_LABEL_OPTIONS = [
    'confirmed_unit_error', 'probable_unit_error', 'valid_but_rare',
    'insufficient_evidence', 'other_parse_issue',
]


# build_hp_quality_dataframe()'in full=pd.concat([train_df, holdout_df], ignore_index=True)
# yapisi POZISYONEL sirayi korur (train_df once, X_full ile AYNI sirada; holdout_df sonra,
# X_holdout ile AYNI sirada) - ama .index etiketlerini SIFIRLAR. Bu yuzden orijinal ozellik
# satirina (predict() icin gereken TUM kolonlarla) POZISYONA gore erisilir, index'e gore DEGIL.
# Yanlislik riskine karsi marka/model/motor_gucu ESLESMESI DOGRULANIR - uyusmazlik olursa
# (guvenlik onlemi) o satir icin tahmin hesaplanmaz, NaN birakilir.
#
# SADECE positions'ta verilen satirlar icin tahmin hesaplar (TUM ~340k satir DEGIL -
# probable_unit_error zaten sadece n=551 kayit, tum veri setinde tek-tek predict()
# cagirmak gereksiz/asiri yavas olurdu).
def build_predictions_for_positions(df, X_full, X_holdout, positions):
    artifact = load_model()
    model = artifact['model']
    n_full = len(X_full)
    preds = pd.Series(np.nan, index=positions, dtype=float)
    mismatch = 0

    for pos in positions:
        row = df.iloc[pos]
        X_row = X_full.iloc[[pos]] if row['split'] == 'egitim' else X_holdout.iloc[[pos - n_full]]
        if (X_row['marka'].iloc[0] != row['marka'] or X_row['model'].iloc[0] != row['model']
                or X_row['motor_gucu'].iloc[0] != row['motor_gucu']):
            mismatch += 1
            continue
        X_aligned = apply_saved_categories(X_row, artifact)
        preds.loc[pos] = model.predict(X_aligned)[0]

    return preds, mismatch


def stratified_sample(df, seed=SAMPLE_SEED):
    df = df.copy()
    df['abs_z'] = df['hp_robust_zscore'].abs()
    try:
        df['tier'] = pd.qcut(df['abs_z'], 3, labels=['low', 'medium', 'high'], duplicates='drop')
    except ValueError:
        df['tier'] = pd.cut(df['abs_z'], 3, labels=['low', 'medium', 'high'])

    rng = np.random.RandomState(seed)

    def sample_tier(tier_name, n, label):
        pool = df[df['tier'] == tier_name]
        n = min(n, len(pool))
        return pool.sample(n=n, random_state=rng).assign(sample_stratum=label)

    high = sample_tier('high', N_HIGH, 'high_score')
    medium = sample_tier('medium', N_MEDIUM, 'medium_score')
    low = sample_tier('low', N_LOW, 'low_score')

    chosen_idx = set(high.index) | set(medium.index) | set(low.index)
    remaining_pool = df[~df.index.isin(chosen_idx)]
    n_random = min(N_RANDOM, len(remaining_pool))
    random_ctrl = remaining_pool.sample(n=n_random, random_state=rng).assign(sample_stratum='random_control')

    return pd.concat([high, medium, low, random_ctrl])


def tier_boundaries_report(df):
    lines = ['tabaka sinirlari (|hp_robust_zscore| tercile kesim noktalari):']
    for tier in ('low', 'medium', 'high'):
        sub = df[df['tier'] == tier]['abs_z']
        if len(sub) == 0:
            lines.append(f'  {tier:8s}: bos (n=0)')
            continue
        lines.append(f'  {tier:8s}: n={len(sub):>4}  |z| araligi=[{sub.min():.2f}, {sub.max():.2f}]  '
                     f'ortalama={sub.mean():.2f}')
    return lines


def discriminant_signal_report(sample):
    """Manuel siniflandirma HENUZ yapilmadigi icin (kasitli olarak bos) 'hangi
    kural gercekten ayirt edici' sorusuna DOGRUDAN cevap veremeyiz - ama tabakalar
    ARASINDA mevcut sinyallerin (kw_hp yakinligi, HP/L fizigi, tahmin hatasi
    buyuklugu) ne kadar FARKLILASTIGINI raporlamak, manuel incelemeyi hangi
    kayitlara/kurallara ODAKLANMASI gerektigi konusunda yonlendirir."""
    lines = ['--- Tabakalar arasi mevcut sinyal farkliliklari (gozlemsel, siniflandirma DEGIL) ---',
             '(NOT: kw_donusumune_cok_yakin TAUTOLOJIK - probable_unit_error etiketinin TANIMI zaten '
             'bu yakinliga dayaniyor, bu yuzden tum tabakalarda ~%100 cikmasi BEKLENEN bir sonuc, '
             'ayirt edici bir sinyal DEGIL; asil ayirt edici olabilecekler |tahmin_hata| buyuklugu ve '
             'z_score_is_mad_zero_sentinel oranidir.)']
    for stratum in ('high_score', 'medium_score', 'low_score', 'random_control'):
        sub = sample[sample['sample_stratum'] == stratum]
        if len(sub) == 0:
            continue
        kw_close = (sub['kw_hp_donusum_yakinligi'] < 0.01).mean() * 100
        hp_l = sub['hp_hacmi_orani']
        implausible_hpl = (hp_l > 350).mean() * 100 if hp_l.notna().any() else float('nan')
        err = sub['tahmin_hata'].dropna()
        mean_abs_err = err.abs().mean() if len(err) > 0 else float('nan')
        lines.append(f'{stratum:16s} n={len(sub):>3}  kw_donusumune_cok_yakin(%<1)=%{kw_close:5.1f}  '
                     f'HP/L>350_fiziksel_supheli=%{implausible_hpl:5.1f}  '
                     f'ort|tahmin_hata|={mean_abs_err:,.0f} TL')
    return lines


def main():
    df, X_full, y_full, X_holdout, y_holdout = build_hp_quality_dataframe()

    unit_error = df[df['tag'] == 'probable_unit_error'].copy()
    total_n = len(unit_error)

    # unit_error.index, df'nin (concat sonrasi) POZISYONEL index'ini korur (boolean
    # filtreleme index sifirlamaz) - bu yuzden dogrudan build_predictions_for_positions'a
    # verilebilir; sadece bu 551 satir icin tahmin hesaplanir, tum ~340k satir DEGIL.
    preds, mismatch = build_predictions_for_positions(df, X_full, X_holdout, unit_error.index)
    unit_error['tahmin'] = preds
    unit_error['tahmin_hata'] = unit_error['tahmin'] - unit_error['fiyat']

    sample = stratified_sample(unit_error)
    sample['kw_hp_donusum_yakinligi'] = sample['hp_deviation_ratio'].apply(
        lambda r: min(abs(r - KW_TO_HP), abs(r - 1 / KW_TO_HP)))
    # hp_robust_zscore, peer grubunun MAD'i TAM SIFIRSA (grup icinde motor_gucu hic
    # degismiyor) +-999 SENTINEL degerine duser (bkz. hp_quality_analysis.robust_zscore) -
    # bu GERCEK bir "asiri anomali" degil, "peer grubu sifir varyansli" edge-case'i.
    # Tercile'a gore "high" tabaka bu satirlarla DOLU olabilir - reviewer'in bunu
    # z'nin surekli/olculebilir bir buyuklugu SANMAMASI icin ayri isaretlenir.
    sample['z_score_is_mad_zero_sentinel'] = sample['hp_peer_mad'] == 0

    out_cols = ['sample_stratum', 'split', 'marka', 'model', 'paket', 'yil', 'motor_hacmi', 'motor_gucu',
                'yakit_turu', 'fiyat', 'tahmin', 'tahmin_hata', 'hp_peer_median', 'hp_peer_mad',
                'hp_peer_count', 'peer_group_type', 'hp_deviation_ratio', 'hp_robust_zscore',
                'z_score_is_mad_zero_sentinel', 'hp_hacmi_orani', 'hp_motor_hacmi_ham_oran',
                'kw_hp_donusum_yakinligi', 'tag']
    sample_out = sample[out_cols].copy()
    sample_out['manual_siniflandirma'] = ''  # BILEREK BOS - bkz. modul docstring, insan doldurur
    sample_out['manual_not'] = ''

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    sample_out.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    lines = ['=== Faz 24: probable_unit_error (n=551) skor-tabakali inceleme ornek kumesi ===']
    lines.append(f'toplam probable_unit_error kaydi: {total_n}')
    lines.append(f'tahmin hizalama uyusmazligi (guvenlik kontrolu, tahmin=NaN birakildi): {mismatch}')
    lines.append('')
    lines.append(f'ornek kumesi: {len(sample_out)} kayit '
                 f'({N_HIGH} yuksek + {N_MEDIUM} orta + {N_LOW} dusuk tabaka + {N_RANDOM} rastgele kontrol)')
    lines.extend(tier_boundaries_report(sample))
    n_sentinel_total = (unit_error['hp_peer_mad'] == 0).sum()
    n_sentinel_sample = int(sample['z_score_is_mad_zero_sentinel'].sum())
    lines.append(f'ONEMLI NOT: 551 kaydin {n_sentinel_total}\'i ({100 * n_sentinel_total / total_n:.0f}%) '
                 f'peer grubunun MAD=0 olmasi nedeniyle |z|=999 SENTINEL degerine sahip (gercek surekli '
                 f'bir anomali buyuklugu DEGIL - "high" tabaka buyuk olcude bu satirlardan olusuyor, '
                 f'ornekte {n_sentinel_sample}/{N_HIGH + N_MEDIUM + N_LOW + N_RANDOM} kayit). Manuel '
                 f'incelemede "high_score" etiketini "en anomalisi" degil "peer grubu sifir varyansli VEYA '
                 f'gercekten cok sapan" olarak okuyun - z_score_is_mad_zero_sentinel sutunu bunu ayirt eder.')
    lines.append('')
    lines.append('manual_siniflandirma icin izin verilen degerler: ' + ', '.join(MANUAL_LABEL_OPTIONS))
    lines.append('BU SUTUN KASITLI OLARAK BOS BIRAKILDI - otomatik siniflandirma YAPILMADI.')
    lines.append('')
    lines.extend(discriminant_signal_report(sample_out.assign(sample_stratum=sample['sample_stratum'].values)))
    lines.append('')
    lines.append('=== Sonraki adim (manuel inceleme TAMAMLANDIKTAN SONRA uygulanacak karar esigi) ===')
    lines.append(
        'manual_siniflandirma doldurulduktan sonra bu 100 kayit uzerinde:\n'
        '  confirmed_unit_error + probable_unit_error orani >= %80 VE belirli bir kural\n'
        '    (orn. kw_hp_donusum_yakinligi<esik) etrafinda yogunlasiyorsa -> otomatik temizlik\n'
        '    kurali gelistirilebilir (bu script bunu YAPMAZ, sadece kanit toplar).\n'
        '  oran %50-80 arasindaysa -> SADECE yuksek-guvenli alt kume (orn. high_score tabakasi +\n'
        '    dusuk kw_hp_donusum_yakinligi) temizlenmeli, geri kalani flag/warning olarak kalir.\n'
        '  oran <%50 ise -> otomatik silme YAPILMAMALI, sadece flag/warning olarak tutulmali.\n'
        'Bu script KARARI VERMEZ - production pipeline\'a veya egitim verisine HICBIR DEGISIKLIK\n'
        'yapilmadi; sadece manuel incelemeyi kolaylastiracak, temsili bir ornek kumesi + baglamsal\n'
        'ozet uretildi.'
    )

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nCSV: {CSV_PATH}')
    print(f'Ozet: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
