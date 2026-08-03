"""Faz 15: model hata taksonomisi ve nadir segment analizi.

Amac: en yuksek hatali tahminleri listelemek degil, TEKRAR EDEN hata
siniflarini nicel olarak belirlemek (frekans-bazli hata artisi, motor gucu
extrapolation, fiyat anomalileri) ve her sinif icin veri-turevli bir karar
uretmek. Dis holdout (cars1_normalized.csv, egitimde hic gorulmemis)
uzerinde calisir - ayni holdout evaluate.py (Faz 11) tarafindan da kullanilir.

SHAP katkilari LightGBM'in native pred_contrib() cikisindan alinir - shap
kutuphanesinin TreeExplainer'iyla matematiksel olarak AYNI degerleri urutur
(dogrulandi: base+sum(contrib)==model.predict(), fark ~1e-8) ama 52.823
satirlik tam holdout icin cok daha hizlidir (~10s vs shap ile dakikalar).

Fiyat anomalisi tespiti (Madde 2) TEK bir sabit esik KULLANMAZ - uc sinyali
birlikte degerlendirir: (1) fiyatin holdout icindeki genel yuzdelik konumu,
(2) egitim verisindeki ayni marka+model icin medyan fiyata oran (peer_ratio),
(3) modelin kendi tahmini/gercek orani (pred_actual_ratio - model peer
ozelliklere gore normal bir fiyat bekliyorsa ama gercek fiyat cok farkliysa,
bu FIYATIN kendisinin anomalisi oldugunu, aracin ozelliklerinin degil,
gucIendirir). cars1_normalized.csv ilan basligi/aciklamasi ICERMEDIGI icin
(bkz. train.py CARS1_EXTRA_COLS) kesin "hurda ilani" / "kapora" gibi metin-
turevli siniflandirma YAPILMAZ - sadece "possible_data_quality_issue" gibi
genel bir isaret konur (bkz. modul ici yorum).
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

from domain_validation import KM_YIL_WARNING_THRESHOLD
from train import load_model, load_cars1_holdout, prepare_external_holdout, prepare_full_training_data

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'error_taxonomy_report.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'error_taxonomy_summary.txt')
PLOT_DIR = os.path.join(BASE_DIR, 'data', 'output', 'error_taxonomy')

PRICE_SEGMENT_EDGES = [0, 150_000, 400_000, 700_000, 1_200_000, 2_000_000, np.inf]
PRICE_SEGMENT_LABELS = ['<150K', '150K-400K', '400K-700K', '700K-1.2M', '1.2M-2M', '2M+']

POWER_EDGES = [0, 75, 100, 150, 200, 300, 400, np.inf]
POWER_LABELS = ['1-75', '76-100', '101-150', '151-200', '201-300', '301-400', '400+']

FREQ_EDGES = [0, 5, 20, 100, 500, 10 ** 9]
FREQ_LABELS = ['1-5', '6-20', '21-100', '101-500', '500+']

OLD_VEHICLE_AGE_THRESHOLD = 35
EXTREME_PEER_RATIO_HIGH = 3.0
EXTREME_PEER_RATIO_LOW = 1 / 3.0
EXTREME_PRED_ACTUAL_RATIO_HIGH = 4.0
EXTREME_PRED_ACTUAL_RATIO_LOW = 0.25


def smape(actual, predicted):
    return 200 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))


def regression_stats(actual, predicted):
    error = predicted - actual
    abs_error = np.abs(error)
    n = len(actual)
    mae = abs_error.mean()
    rmse = np.sqrt((error ** 2).mean())
    bias = error.mean()
    medae = np.median(abs_error)
    smape_val = smape(actual, predicted).mean()
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'n': n, 'mae': mae, 'rmse': rmse, 'bias': bias, 'medae': medae, 'smape': smape_val, 'r2': r2}


def format_stats_row(label, stats):
    r2_str = f'{stats["r2"]:.4f}' if not np.isnan(stats['r2']) else 'n/a'
    return (f'{label:14s} n={stats["n"]:>6,} MAE={stats["mae"]:>10,.0f} RMSE={stats["rmse"]:>10,.0f} '
            f'bias={stats["bias"]:>+10,.0f} MedAE={stats["medae"]:>9,.0f} sMAPE={stats["smape"]:>6.1f}% '
            f'R2={r2_str}')


def build_error_dataframe():
    artifact = load_model()
    model = artifact['model']

    X_full, y_full = prepare_full_training_data()
    X_holdout, y_holdout = prepare_external_holdout(X_full)

    preds = model.predict(X_holdout)
    contribs = model.booster_.predict(X_holdout, pred_contrib=True)
    feature_names = list(X_holdout.columns)

    actual = y_holdout.values
    error = preds - actual
    abs_error = np.abs(error)
    pct_error = error / actual * 100
    smape_val = smape(actual, preds)
    # LightGBM bazen negatif tahmin uretebilir (bkz. Hyundai Accent 1997,
    # motor_gucu=601 HP - acik bir veri hatasi - ornegi, summary'de not
    # edildi). log1p(negatif<-1) tanimsiz oldugu icin 0'a kirpiliyor; bu
    # satirlar zaten mutlak_hata/tags ile ayrica goruluyor, log_error'da
    # NaN birakmak yerine "asiri" bir deger (0 fiyat gibi ele alinir) vermek
    # ortalama hesaplarini bozmaz.
    log_error = np.log1p(np.clip(preds, 0, None)) - np.log1p(actual)

    df = pd.DataFrame({
        'gercek_fiyat': actual,
        'tahmin': preds,
        'hata': error,
        'mutlak_hata': abs_error,
        'yuzde_hata': pct_error,
        'smape': smape_val,
        'log_error': log_error,
        'marka': X_holdout['marka'].astype(str).values,
        'model': X_holdout['model'].astype(str).values,
        'paket': X_holdout['paket'].astype(str).values,
        'yil': X_holdout['yil'].values,
        'kilometre': X_holdout['kilometre'].values,
        'motor_gucu': X_holdout['motor_gucu'].values,
        'agir_hasarli': X_holdout['agir_hasarli'].values,
        'yas': X_holdout['yas'].values,
        'km_yil': X_holdout['km_yil'].values,
    })

    df['fiyat_segmenti'] = pd.cut(df['gercek_fiyat'], PRICE_SEGMENT_EDGES, labels=PRICE_SEGMENT_LABELS)
    df['motor_gucu_dilimi'] = pd.cut(df['motor_gucu'], POWER_EDGES, labels=POWER_LABELS)

    # egitim frekanslari (X_full - modelin GERCEKTEN gordugu veri)
    model_freq_map = X_full['model'].value_counts()
    paket_freq_map = X_full['paket'].value_counts()
    df['model_egitim_frekansi'] = df['model'].map(model_freq_map).fillna(0).astype(int)
    df['paket_egitim_frekansi'] = df['paket'].map(paket_freq_map).fillna(0).astype(int)
    df['model_frekans_grubu'] = pd.cut(df['model_egitim_frekansi'], FREQ_EDGES, labels=FREQ_LABELS)
    df['paket_frekans_grubu'] = pd.cut(df['paket_egitim_frekansi'], FREQ_EDGES, labels=FREQ_LABELS)

    # marka+model peer medyani (egitim verisinden) - fiyat anomalisi tespiti icin
    train_prices = X_full.assign(fiyat=y_full.values)
    peer_median = train_prices.groupby(['marka', 'model'], observed=True)['fiyat'].median()
    brand_median = train_prices.groupby('marka', observed=True)['fiyat'].median()
    overall_median = y_full.median()

    def lookup_peer_median(row):
        key = (row['marka'], row['model'])
        if key in peer_median.index:
            return peer_median.loc[key]
        if row['marka'] in brand_median.index:
            return brand_median.loc[row['marka']]
        return overall_median

    df['peer_medyan_fiyat'] = df.apply(lookup_peer_median, axis=1)
    df['peer_orani'] = df['gercek_fiyat'] / df['peer_medyan_fiyat']
    df['tahmin_gercek_orani'] = df['tahmin'] / df['gercek_fiyat']
    df['fiyat_yuzdelik_konumu'] = df['gercek_fiyat'].rank(pct=True) * 100

    # SHAP (pred_contrib) - en baskin ilk 3 ozellik
    top3_list = []
    for i in range(len(df)):
        row_contribs = contribs[i, :-1]  # son kolon base_value
        order = np.argsort(-np.abs(row_contribs))[:3]
        parts = [f'{feature_names[j]}:{row_contribs[j]:+.0f}' for j in order]
        top3_list.append(';'.join(parts))
    df['shap_top3'] = top3_list

    df['tags'] = df.apply(lambda r: ';'.join(compute_tags(r)), axis=1)

    return df, X_full, y_full


def compute_tags(row):
    tags = []
    if row['model_frekans_grubu'] in ('1-5', '6-20'):
        tags.append('rare_model')
    if row['paket_frekans_grubu'] in ('1-5', '6-20'):
        tags.append('rare_package')
    if row['motor_gucu'] > 300:
        tags.append('high_power_extrapolation')
    if row['km_yil'] > KM_YIL_WARNING_THRESHOLD:
        tags.append('high_mileage_outlier')
    if row['yas'] > OLD_VEHICLE_AGE_THRESHOLD:
        tags.append('old_vehicle_outlier')

    is_peer_low = row['peer_orani'] < EXTREME_PEER_RATIO_LOW
    is_peer_high = row['peer_orani'] > EXTREME_PEER_RATIO_HIGH
    is_pred_extreme = (row['tahmin_gercek_orani'] > EXTREME_PRED_ACTUAL_RATIO_HIGH or
                        row['tahmin_gercek_orani'] < EXTREME_PRED_ACTUAL_RATIO_LOW)
    if is_peer_low or is_peer_high:
        if is_pred_extreme:
            # hem peer-grubuna gore hem modelin kendi beklentisine gore asiri sapma -
            # metin (ilan basligi/aciklamasi) yok, kesin siniflandirma yapilamaz.
            tags.append('possible_data_quality_issue')
        elif is_peer_low:
            tags.append('price_anomaly_low')
        else:
            tags.append('price_anomaly_high')

    if not tags:
        tags.append('ordinary_model_error')
    return tags


def freq_bucket_table(df, freq_col, label):
    lines = [f'=== {label} frekans grubuna gore performans ===']
    for bucket in FREQ_LABELS:
        sub = df[df[freq_col] == bucket]
        if len(sub) == 0:
            lines.append(f'{bucket:14s} n=0')
            continue
        stats = regression_stats(sub['gercek_fiyat'].values, sub['tahmin'].values)
        lines.append(format_stats_row(bucket, stats))
    return lines


def power_bin_table(df):
    lines = ['=== motor gucu dilimine gore performans ===']
    for bucket in POWER_LABELS:
        sub = df[df['motor_gucu_dilimi'] == bucket]
        if len(sub) == 0:
            lines.append(f'{bucket:14s} n=0')
            continue
        stats = regression_stats(sub['gercek_fiyat'].values, sub['tahmin'].values)
        gercek_ort = sub['gercek_fiyat'].mean()
        tahmin_ort = sub['tahmin'].mean()
        lines.append(format_stats_row(bucket, stats) +
                     f'  gercek_ort={gercek_ort:,.0f} tahmin_ort={tahmin_ort:,.0f}')
    return lines


def model_power_cross_examples(df, top_n=10):
    """Ayni HP diliminde farkli modellerin ne kadar farkli fiyatlandigini
    gosteren ornekler - 'ayni HP farkli anlam' bulgusu icin."""
    lines = ['=== model x motor_gucu capraz ornekleri (ayni HP dilimi, farkli fiyat davranisi) ===']
    for bucket in ['201-300', '301-400', '400+']:
        sub = df[df['motor_gucu_dilimi'] == bucket]
        if len(sub) < 5:
            continue
        grp = sub.groupby('model', observed=True).agg(
            n=('gercek_fiyat', 'size'),
            gercek_ort=('gercek_fiyat', 'mean'),
            tahmin_ort=('tahmin', 'mean'),
            bias=('hata', 'mean'),
        ).query('n >= 3').sort_values('bias', key=abs, ascending=False)
        if len(grp) == 0:
            continue
        lines.append(f'-- HP dilimi {bucket} --')
        for model_name, row in grp.head(5).iterrows():
            lines.append(f'  {model_name:20s} n={row["n"]:>3.0f} gercek_ort={row["gercek_ort"]:>10,.0f} '
                         f'tahmin_ort={row["tahmin_ort"]:>10,.0f} bias={row["bias"]:>+10,.0f}')
    return lines


def bmw_m_serisi_deepdive(df, X_full):
    lines = ['=== BMW M Serisi derinlemesine inceleme ===']
    rows = df[(df['marka'] == 'BMW') & (df['model'] == 'M Serisi')]
    if len(rows) == 0:
        lines.append('holdout icinde BMW M Serisi kaydi yok.')
        return lines
    model_freq = (X_full['model'] == 'M Serisi').sum()
    paket_freq_counts = X_full[X_full['model'] == 'M Serisi']['paket'].value_counts()
    power_percentile = (X_full['motor_gucu'] <= rows['motor_gucu'].iloc[0]).mean() * 100

    lines.append(f'holdout kayit sayisi: {len(rows)}')
    lines.append(f'"M Serisi" model egitim frekansi (tum markalar dahil model adi ayni olabilir): {model_freq}')
    lines.append(f'paket dagilimi (ilk 5): {paket_freq_counts.head(5).to_dict()}')
    lines.append(f'motor_gucu={rows["motor_gucu"].iloc[0]:.0f} HP -> egitim verisinde yuzdelik konum: %{power_percentile:.2f}')
    for _, r in rows.iterrows():
        lines.append(f'  paket={r["paket"]!r} yil={r["yil"]:.0f} km={r["kilometre"]:,.0f} '
                     f'gercek={r["gercek_fiyat"]:,.0f} tahmin={r["tahmin"]:,.0f} hata={r["hata"]:+,.0f} '
                     f'model_freq_grubu={r["model_frekans_grubu"]} paket_freq_grubu={r["paket_frekans_grubu"]} '
                     f'shap_top3={r["shap_top3"]}')
    return lines


def negative_prediction_check(df):
    """serve.py /predict, pred<=0 durumunda 502 dondurur (bkz. Faz 12) - bu
    holdout'ta modelin GERCEKTEN negatif tahmin urettigi satirlari bularak o
    korumanin neden gerekli oldugunu somut bir ornekle dogrular."""
    lines = ['=== Negatif/sifir tahmin kontrolu (serve.py 502 korumasinin gerekce ornegi) ===']
    bad = df[df['tahmin'] <= 0]
    lines.append(f'tahmin<=0 olan satir sayisi: {len(bad)} / {len(df):,}')
    for _, r in bad.iterrows():
        lines.append(f'  {r["marka"]} {r["model"]} {r["paket"]!r} yil={r["yil"]:.0f} '
                     f'motor_gucu={r["motor_gucu"]:.0f} HP gercek={r["gercek_fiyat"]:,.0f} '
                     f'tahmin={r["tahmin"]:,.0f} -> motor_gucu bu deger icin gercek disi '
                     f'(egitim verisinde ayni marka/model icin tipik guc cok daha dusuk); '
                     f'muhtemel veri girisi hatasi.')
    return lines


def decision_table(df):
    def tag_stats(tag):
        sub = df[df['tags'].str.contains(tag)]
        n = len(sub)
        return n, 100 * n / len(df), sub['mutlak_hata'].mean(), sub['hata'].mean(), sub['smape'].mean()

    rows = [
        ('Fiyat anomalisi (dusuk)', 'price_anomaly_low',
         'Gercekten cok ucuz satilan araclar (eski/yuksek km) VEYA hurda/kapora/on odeme '
         'gibi fiyat-disi bir kayit - metin (ilan basligi/aciklamasi) olmadigi icin kesin '
         'ayrim yapilamiyor.',
         'Uretim formunda gercekci-olmayan dusuk fiyat girisi icin ayri bir uyari; '
         'yeniden kazima kapsamina ilan basligi/aciklamasi eklenmesi.'),
        ('Fiyat anomalisi (yuksek)', 'price_anomaly_high',
         'Koleksiyon/nadir/ozel arac fiyat primi - model bunu ozelliklerden yakalayamiyor.',
         'Cok kucuk n - simdilik dusuk oncelik, izlemeye devam.'),
        ('Nadir model (egitim freq<=20)', 'rare_model',
         'Yetersiz egitim ornegi - modelin o model icin ogrenecegi ozgul fiyat sinyali yok.',
         'bkz. Madde 8 karari (marka-model hiyerarsik fallback).'),
        ('Nadir paket (egitim freq<=20)', 'rare_package',
         'Serbest metin paket alaninin dogal sonucu (6.444 farkli deger, %47si freq<=5) - '
         'ANCAK asagidaki frekans tablosunda hata frekansla GUCLU korele DEGIL.',
         'bkz. Madde 8 karari (autocomplete/canonical matching, agir encoding DEGIL).'),
        ('Yuksek guc extrapolation (>300HP)', 'high_power_extrapolation',
         'Egitim P99.99=601 HP sinirinin otesinde ekstrapolasyon + bazi kayitlarda '
         'muhtemel veri girisi hatasi (bkz. Hyundai Accent 601HP ornegi).',
         'motor_gucu ust sinirinin uzerindeki (mevcut domain_validation.py: 800) '
         'girdilerde ayrica uyari; egitim verisinde motor_gucu>800 olan kayitlarin '
         'veri kalitesi denetimi.'),
        ('Eski arac (yas>35)', 'old_vehicle_outlier',
         'Az ornek + eski araclarda fiyat-yas iliskisi dogrusal degil (koleksiyon '
         'degeri bazen yasla ARTAR).',
         'Dusuk-orta oncelik, ayrica izlenmeli.'),
        ('Olasi veri kalitesi sorunu', 'possible_data_quality_issue',
         'Hem peer-gruba hem modelin kendi beklentisine gore asiri sapma - kayit hatasi '
         'ihtimali yuksek.',
         'Kucuk n (10) ama COK yuksek etki (sMAPE %149) - veri denetimi ONCELIGI YUKSEK.'),
    ]

    lines = ['=== Karar tablosu (Madde 7) ===',
             f'{"Hata sinifi":32s} {"Yayginlik":>16s} {"MAE":>12s} {"bias":>12s} {"sMAPE":>8s}  Muhtemel neden / Onerilen cozum']
    for label, tag, cause, fix in rows:
        n, pct, mae, bias, smape_val = tag_stats(tag)
        lines.append(f'{label:32s} n={n:>5,} (%{pct:4.1f}) {mae:>12,.0f} {bias:>+12,.0f} %{smape_val:>6.1f}')
        lines.append(f'    neden: {cause}')
        lines.append(f'    cozum: {fix}')
    return lines


def paket_model_strategy_decision(df):
    model_freq_stats = {b: df[df['model_frekans_grubu'] == b]['mutlak_hata'].mean() for b in FREQ_LABELS}
    paket_freq_stats = {b: df[df['paket_frekans_grubu'] == b]['mutlak_hata'].mean() for b in FREQ_LABELS}

    model_ratio = model_freq_stats['1-5'] / model_freq_stats['500+']
    paket_ratio = paket_freq_stats['1-5'] / paket_freq_stats['500+']
    rare_paket_pct = 100 * (df['paket_frekans_grubu'].isin(['1-5', '6-20'])).mean()
    rare_model_pct = 100 * (df['model_frekans_grubu'].isin(['1-5', '6-20'])).mean()

    lines = ['=== Madde 8: paket/model stratejisi kararı (veriye dayali) ===',
             f'model MAE orani (freq 1-5 / freq 500+): {model_ratio:.2f}x -> GUCLU frekans-hata korelasyonu',
             f'paket MAE orani (freq 1-5 / freq 500+): {paket_ratio:.2f}x -> ZAYIF/monoton-olmayan frekans-hata iliskisi',
             f'holdout satirlarinin %{rare_model_pct:.1f}i nadir model (freq<=20), %{rare_paket_pct:.1f}i nadir paket (freq<=20)',
             '',
             'KARAR - MODEL alani icin:',
             '  Nadir modellerde hata gercekten sistematik olarak yukseliyor (yaklasik '
             f'{model_ratio:.1f}x) ama etkilenen satir orani kucuk (%{rare_model_pct:.1f}). '
             'Buyuk bir mimari degisiklik (target encoding, agir hiyerarsik ozellik '
             'muhendisligi) bu kucuk kazanc icin orantisiz risk/karmasiklik getirir. '
             'Onerilen: "marka-model hiyerarsik ozellikler" (secenek 6) - HAFIF bir eklenti '
             'olarak: model_egitim_frekansi ve/veya marka-seviyesi medyan fiyat gibi 1-2 '
             'ek sayisal ozellik egitime eklenebilir (LightGBM zaten native missing '
             'destekliyor, mevcut kategorik yaklasim BOZULMAZ - secenek 1 esasen korunur, '
             'uzerine ince bir katman eklenir).',
             '',
             'KARAR - PAKET alani icin:',
             '  Nadirlik ile hata arasinda net bir iliski YOK (paket_ratio '
             f'{paket_ratio:.2f}x, model_ratio {model_ratio:.2f}x ile karsilastirilinca cok '
             'zayif) - bu, paket\'in asil sorununun MODEL DOGRULUGU degil, INFERENCE-TIME '
             'ESLESME (serbest metin -> egitim kelime hazinesi) oldugunu gosteriyor. '
             'Onerilen: "model-bagimli paket dropdown" + "autocomplete + canonical '
             'matching" (secenek 3+4, category_mapping.py/Faz 13 desenini paket icin '
             'genisletmek) - kullanicinin secilen modele gore bilinen paket listesinden '
             'secmesi/otomatik tamamlamasi, boylece serbest metnin egitim kelime '
             'hazinesiyle eslesme orani artar. "Other altinda birlestirme" (secenek 2) '
             'ONERILMEZ - paket zaten dusuk SHAP etkili (bkz. Faz 11 SHAP raporu, ~%5.6 '
             'gain), rare paketleri "Other"a toplamak var olan zayif sinyali daha da '
             'sulandirir, hatayi olcumlenebilir sekilde AZALTMAZ (frekans-hata korelasyonu '
             'zaten zayif oldugu icin).',
             '',
             'KARAR - "yeniden egitimde bilinmeyen kategori stratejisi" (secenek 7):',
             '  Zaten dogru sekilde uygulaniyor - LightGBM native kategori + missing '
             'destegi (train.py to_category/apply_saved_categories), gorulmemis bir '
             'kategoriyi NaN olarak ele aliyor. Bu, formalize edilip DOKUMANTE edilmeli '
             '(bu rapor + train.py yorumlari) ama KOD DEGISIKLIGI gerektirmiyor.',
             '',
             'REDDEDILEN secenekler ve gerekce:',
             '  - "target/frequency encoding" (secenek 5): LightGBM native kategorik '
             'zaten benzer bir sizinti-onlemeli davranis sagliyor; ek karmasiklik/sizinti '
             'riski gozlemlenen etki buyuklugune (ozellikle paket icin) gore haklandirilmiyor.',
             '  - "dusuk frekanslilari Other altinda birlestirme" (secenek 2): model icin '
             'de onerilmiyor - nadir modeller arasinda GERCEK fiyat farklari var (bkz. '
             'model x motor_gucu capraz tablosu), "Other" bunu ortadan kaldirir.']
    return lines


def make_plots(df):
    os.makedirs(PLOT_DIR, exist_ok=True)
    paths = []

    def save(fig, name):
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        paths.append(path)

    # 1. model frekansina gore MAE
    fig, ax = plt.subplots(figsize=(7, 4))
    grp = df.groupby('model_frekans_grubu', observed=True)['mutlak_hata'].mean().reindex(FREQ_LABELS)
    ax.bar(FREQ_LABELS, grp.values)
    ax.set_xlabel('model egitim frekans grubu')
    ax.set_ylabel('MAE (TL)')
    ax.set_title('Model frekansina gore MAE')
    save(fig, 'mae_by_model_freq.png')

    # 2. paket frekansina gore MAE
    fig, ax = plt.subplots(figsize=(7, 4))
    grp = df.groupby('paket_frekans_grubu', observed=True)['mutlak_hata'].mean().reindex(FREQ_LABELS)
    ax.bar(FREQ_LABELS, grp.values)
    ax.set_xlabel('paket egitim frekans grubu')
    ax.set_ylabel('MAE (TL)')
    ax.set_title('Paket frekansina gore MAE')
    save(fig, 'mae_by_paket_freq.png')

    # 3. motor gucu dilimine gore bias
    fig, ax = plt.subplots(figsize=(7, 4))
    grp = df.groupby('motor_gucu_dilimi', observed=True)['hata'].mean().reindex(POWER_LABELS)
    colors = ['tab:red' if v < 0 else 'tab:blue' for v in grp.values]
    ax.bar(POWER_LABELS, grp.values, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('motor gucu dilimi (HP)')
    ax.set_ylabel('bias = tahmin - gercek (TL)')
    ax.set_title('Motor gucu dilimine gore bias')
    save(fig, 'bias_by_power_bin.png')

    # 4. gercek fiyat vs tahmin
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['gercek_fiyat'], df['tahmin'], s=3, alpha=0.15)
    lims = [0, max(df['gercek_fiyat'].max(), df['tahmin'].max())]
    ax.plot(lims, lims, color='red', linewidth=1)
    ax.set_xlabel('gercek fiyat (TL)')
    ax.set_ylabel('tahmin (TL)')
    ax.set_title('Gercek fiyat vs tahmin')
    save(fig, 'actual_vs_predicted.png')

    # 5. yuzde hata vs gercek fiyat
    fig, ax = plt.subplots(figsize=(7, 5))
    clipped = df['yuzde_hata'].clip(-200, 200)
    ax.scatter(df['gercek_fiyat'], clipped, s=3, alpha=0.15)
    ax.axhline(0, color='red', linewidth=1)
    ax.set_xlabel('gercek fiyat (TL)')
    ax.set_ylabel('yuzde hata (%) [-200,200 kirpilmis]')
    ax.set_title('Yuzde hata vs gercek fiyat')
    save(fig, 'pct_error_vs_price.png')

    # 6. en yuksek 20 pozitif ve negatif hata
    top_pos = df.nlargest(20, 'hata')
    top_neg = df.nsmallest(20, 'hata')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels_pos = [f'{m} {mo}'[:22] for m, mo in zip(top_pos['marka'], top_pos['model'])]
    labels_neg = [f'{m} {mo}'[:22] for m, mo in zip(top_neg['marka'], top_neg['model'])]
    axes[0].barh(labels_pos[::-1], top_pos['hata'][::-1], color='tab:blue')
    axes[0].set_title('En yuksek 20 pozitif hata (fazla tahmin)')
    axes[0].set_xlabel('hata (TL)')
    axes[1].barh(labels_neg[::-1], top_neg['hata'][::-1], color='tab:red')
    axes[1].set_title('En yuksek 20 negatif hata (az tahmin)')
    axes[1].set_xlabel('hata (TL)')
    fig.tight_layout()
    save(fig, 'top20_errors.png')

    return paths


def main():
    df, X_full, y_full = build_error_dataframe()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    report = []
    report.append('=== Genel performans (dis holdout, n={:,}) ==='.format(len(df)))
    overall = regression_stats(df['gercek_fiyat'].values, df['tahmin'].values)
    report.append(format_stats_row('TUMU', overall))
    report.append(f'ortalama sMAPE: %{df["smape"].mean():.2f}, medyan sMAPE: %{df["smape"].median():.2f}')
    report.append(f'ortalama |log_error|: {df["log_error"].abs().mean():.4f}')
    report.append('')

    report.extend(freq_bucket_table(df, 'model_frekans_grubu', 'Model'))
    report.append('')
    report.extend(freq_bucket_table(df, 'paket_frekans_grubu', 'Paket'))
    report.append('')

    report.extend(power_bin_table(df))
    report.append('')
    report.extend(model_power_cross_examples(df))
    report.append('')
    report.extend(bmw_m_serisi_deepdive(df, X_full))
    report.append('')
    report.extend(negative_prediction_check(df))
    report.append('')

    report.append('=== Etiket (tag) dagilimi ve ortalama hata ===')
    for tag in ['price_anomaly_low', 'price_anomaly_high', 'rare_model', 'rare_package',
                'high_power_extrapolation', 'high_mileage_outlier', 'old_vehicle_outlier',
                'possible_data_quality_issue', 'ordinary_model_error']:
        mask = df['tags'].str.contains(tag)
        n = mask.sum()
        if n == 0:
            report.append(f'{tag:28s} n=0')
            continue
        sub = df[mask]
        report.append(f'{tag:28s} n={n:>6,} (%{100*n/len(df):5.1f})  '
                      f'MAE={sub["mutlak_hata"].mean():>10,.0f}  bias={sub["hata"].mean():>+10,.0f}  '
                      f'sMAPE=%{sub["smape"].mean():.1f}')
    report.append('')

    plot_paths = make_plots(df)
    report.append('=== Kaydedilen grafikler ===')
    report.extend(plot_paths)
    report.append('')

    report.extend(decision_table(df))
    report.append('')
    report.extend(paket_model_strategy_decision(df))
    report.append('')

    report.append(f'rapor (satir bazli): {CSV_PATH}')

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f'\nozet kaydedildi: {SUMMARY_PATH}')
    print(f'satir-bazli CSV kaydedildi: {CSV_PATH}')


if __name__ == '__main__':
    main()
