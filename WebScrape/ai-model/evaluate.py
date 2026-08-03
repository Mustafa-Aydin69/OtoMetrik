"""Faz 11: train.py (Faz 10) tarafindan kaydedilen final modelin dis holdout
(cars1_normalized.csv) uzerinde degerlendirilmesi, yorumlanabilirlik (SHAP) katmani ve
sonuclarin guvenilirligini sorgulayan destekleyici kontroller.

Faz 11 Madde 1: dis holdout metrikleri (train.py main()'de zaten hesaplaniyordu, burada
tekrar uretilebilir/rapor edilebilir hale getirildi).

Faz 11 Madde 2: segment bazli performans. agir_hasarli, class-imbalance nedeniyle global
feature importance'ta dusuk cikiyor (bkz. Madde 4 SHAP) - bu da'nin gercekten sinyal
tasiyip tasimadigini/modelin bu segmentte sistematik yanli olup olmadigini ayri MAE/RMSE
ve ortalama rezidu (pred - gercek) ile test eder.

Faz 11 Madde 3: cross-source duplicate/leakage kontrolu. cars1_normalized.csv semasi
(seri/model, tramer, cekis, yakit_deposu alanlari) arabam.com'un kendi ilan detay
tablosuna cok benziyor - cars1'in kismen ayni siteden farkli bir tarihte cekilmis olma
ihtimali var. marka+model+yil+kilometre+fiyat'in TAM esletigi holdout satirlari
"supheli duplicate" olarak isaretlenir; bu satirlar cikarilinca R2/MAE/RMSE'nin ne kadar
degistigi olculerek headline metriginin bu ortusmeden ne kadar sisirildigi tahmin edilir.

Faz 11 Madde 4: SHAP yorumlanabilirlik katmani. Feature importance (split/gain) tek basina
"model ezberlemiyor" iddiasini kanitlamaz - sadece modelin piyasa mantigiyla uyumlu
degiskenlere dayandigini ve tahminlerin aciklanabilir oldugunu gosterir. Bu bolum global
siralama (mean |SHAP| in TL), summary plot, kritik degiskenler icin dependence plot, yerel
aciklama ornekleri (dusuk/orta/yuksek fiyat, yuksek hata, agir_hasarli) ve otomatik bir
toplam-tutarlilik testi (base_value + shap_values.sum() == prediction) uretir.

Faz 11 Madde 5: izlenebilirlik metadata (model_version, dataset_hash, evaluation_date,
shap sample size, random_seed) - sonuclarin hangi model/veri durumuna ait oldugu her
rapor calistirmasinda acikca kayit altina alinir.
"""
import hashlib
import os
import sys
from datetime import date

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train import (
    MODEL_PATH,
    load_cars1_holdout,
    load_model,
    prepare_external_holdout,
    prepare_full_training_data,
)

SHAP_SAMPLE_SIZE = 8000
RANDOM_SEED = 42

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv')
REPORT_DIR = os.path.join(BASE_DIR, 'data', 'output', 'eval_report')
REPORT_TXT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'eval_report.txt')

DUPLICATE_KEY_COLS = ['marka', 'model', 'yil', 'kilometre', 'fiyat']
DEPENDENCE_PLOT_COLS = ['yil', 'kilometre', 'motor_gucu', 'boyali_sayisi', 'agir_hasarli']


def _md5_of_file(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_run_metadata():
    return {
        'model_version': _md5_of_file(MODEL_PATH)[:12],
        'dataset_hash': _md5_of_file(TRAIN_PATH)[:12],
        'evaluation_date': date.today().isoformat(),
        'shap_sample_size': SHAP_SAMPLE_SIZE,
        'random_seed': RANDOM_SEED,
    }


def evaluate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    bias = np.mean(y_pred - y_true)
    return {'n': len(y_true), 'mae': mae, 'rmse': rmse, 'r2': r2, 'bias': bias}


def format_metrics(label, m):
    return (f'{label}: n={m["n"]:,} MAE={m["mae"]:,.0f} RMSE={m["rmse"]:,.0f} '
            f'R2={m["r2"]:.4f} bias(pred-gercek)={m["bias"]:+,.0f}')


# Madde 2: agir_hasarli=1 vs 0 segmentlerinde ayri MAE/RMSE/bias - dusuk global importance'in
# class-imbalance'tan mi (segment kendi icinde tutarli, sadece az veri) yoksa modelin bu
# segmentte sistematik yanli tahmin uretmesinden mi kaynaklandigini ayirt eder. cars1'de
# agir_hasarli alani hic yok (train.py CARS1_MISSING_COLS) - holdout'ta agir_hasarli=1 orneği
# olmadigi icin bu segment SADECE egitim verisinde (in-sample, gercek out-of-sample degil)
# olculebilir; bu, gercek genelleme performansi degil, yalnizca modelin bu segmente ne kadar
# farkli/yanli tahmin urettigine dair kaba bir isarettir.
def segment_performance(X_holdout, y_holdout, preds, X_full=None, y_full=None, model=None):
    lines = []
    for value, label in [(0, 'agir_hasarli=Hayir'), (1, 'agir_hasarli=Evet')]:
        mask = (X_holdout['agir_hasarli'] == value).values
        if mask.sum() == 0:
            lines.append(f'{label}: dis holdout icinde ornek yok (cars1de bu alan yok)')
            continue
        m = evaluate_metrics(y_holdout[mask].values, preds[mask])
        lines.append(format_metrics(label, m))

    if X_full is not None and (X_holdout['agir_hasarli'] == 1).sum() == 0:
        lines.append('-- asagidaki iki satir IN-SAMPLE (egitim verisinin kendisi uzerinde, '
                      'gercek out-of-sample DEGIL) - sadece yonelim/yanlilik sinyali icin --')
        preds_full = model.predict(X_full)
        for value, label in [(0, 'agir_hasarli=Hayir (in-sample)'), (1, 'agir_hasarli=Evet (in-sample)')]:
            mask = (X_full['agir_hasarli'] == value).values
            if mask.sum() == 0:
                continue
            m = evaluate_metrics(y_full.values[mask], preds_full[mask])
            lines.append(format_metrics(label, m))
    return lines


# Madde 3: marka+model+yil+km+fiyat TAM esleyen holdout satirlari "supheli duplicate"
# olarak isaretlenir (farkli kaynaklarda ayni aracin/ilanin bulunma ihtimali). Bu satirlar
# cikarilinca metriklerin ne kadar degistigi, headline R2'nin ortusmeden ne kadar
# sisirildigini gosterir.
def duplicate_leakage_check(train_df, holdout_df, y_holdout, preds):
    t = train_df[DUPLICATE_KEY_COLS].dropna()
    train_keys = set(zip(t['marka'].str.lower(), t['model'].str.lower(),
                          t['yil'], t['kilometre'], t['fiyat']))

    h_keys = list(zip(holdout_df['marka'].str.lower(), holdout_df['model'].str.lower(),
                       holdout_df['yil'], holdout_df['kilometre'], holdout_df['fiyat']))
    is_dupe = np.array([k in train_keys for k in h_keys])

    lines = [f'supheli duplicate (marka+model+yil+km+fiyat train ile tam esleyen): '
             f'{is_dupe.sum():,} / {len(holdout_df):,} (%{100 * is_dupe.mean():.2f})']

    m_all = evaluate_metrics(y_holdout.values, preds)
    lines.append(format_metrics('tum holdout', m_all))

    if is_dupe.sum() > 0 and (~is_dupe).sum() > 0:
        m_clean = evaluate_metrics(y_holdout.values[~is_dupe], preds[~is_dupe])
        lines.append(format_metrics('duplicate haric', m_clean))
        m_dupe = evaluate_metrics(y_holdout.values[is_dupe], preds[is_dupe])
        lines.append(format_metrics('sadece duplicate', m_dupe))
        lines.append(f'duplicate haric R2 farki: {m_clean["r2"] - m_all["r2"]:+.4f} '
                      '(kucukse headline metrik ortusmeden sisirilmemis demektir)')
    return lines


# Madde 4: global SHAP siralamasi (mean |SHAP| TL), top-N. Gain/split importance'in yuksek
# kardinaliteli kategorik sutunlari sisirebilecegi bilindigi icin (marka/model/paket), SHAP
# gercek marjinal katkiyi olcerek daha guvenilir bir siralama verir.
def global_shap_ranking(shap_values, sample, top_n=20):
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns)
    mean_abs = mean_abs.sort_values(ascending=False)
    lines = [f'{name:30s} {val:12,.0f} TL' for name, val in mean_abs.head(top_n).items()]
    return lines


def save_summary_plot(shap_values, sample):
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    path = os.path.join(REPORT_DIR, 'shap_summary.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    return path


def save_dependence_plots(shap_values, sample, columns=DEPENDENCE_PLOT_COLS):
    os.makedirs(REPORT_DIR, exist_ok=True)
    paths = []
    for col in columns:
        plt.figure()
        shap.dependence_plot(col, shap_values, sample, interaction_index=None, show=False)
        path = os.path.join(REPORT_DIR, f'shap_dependence_{col}.png')
        plt.savefig(path, bbox_inches='tight', dpi=120)
        plt.close()
        paths.append(path)
    return paths


# Madde 4: "bu arac icin fiyat neden bu sekilde tahmin edildi" tarzi yerel aciklamalar.
# Dis holdout'ta agir_hasarli her zaman 0 (cars1'de bu alan yok, bkz. train.py
# CARS1_MISSING_COLS) - o yuzden agir_hasarli ornegi egitim verisinden secilir, digerleri
# (dusuk/orta/yuksek fiyat, yuksek hata) gercek/hic gorulmemis holdout'tan secilir.
def local_explanations(explainer, X_holdout, y_holdout, preds, X_full):
    lines = []
    y_arr = y_holdout.values
    idx_low = int(np.argmin(y_arr))
    idx_high = int(np.argmax(y_arr))
    idx_mid = int(np.argmin(np.abs(y_arr - np.median(y_arr))))
    idx_err = int(np.argmax(np.abs(preds - y_arr)))

    cases = [
        ('dusuk fiyatli arac', X_holdout.iloc[[idx_low]], y_arr[idx_low]),
        ('orta segment arac', X_holdout.iloc[[idx_mid]], y_arr[idx_mid]),
        ('yuksek fiyatli arac', X_holdout.iloc[[idx_high]], y_arr[idx_high]),
        ('en yuksek hatali tahmin', X_holdout.iloc[[idx_err]], y_arr[idx_err]),
    ]

    agir_hasarli_rows = X_full[X_full['agir_hasarli'] == 1]
    if len(agir_hasarli_rows) > 0:
        sample_row = agir_hasarli_rows.sample(1, random_state=RANDOM_SEED)
        cases.append(('agir hasarli arac (egitim verisinden - holdout bu alani icermiyor)',
                      sample_row, None))

    for label, row_df, actual in cases:
        sv = explainer.shap_values(row_df)[0]
        base = explainer.expected_value
        pred = base + sv.sum()
        contrib = pd.Series(sv, index=row_df.columns).sort_values(key=np.abs, ascending=False)
        top = contrib.head(6)
        rest = contrib.iloc[6:].sum()

        lines.append(f'--- {label} ---')
        desc = ', '.join(f'{c}={row_df.iloc[0][c]}' for c in
                          ['marka', 'model', 'yil', 'kilometre', 'vites', 'motor_gucu'])
        lines.append(f'  {desc}')
        lines.append(f'  baz deger: {base:,.0f} TL')
        for name, val in top.items():
            sign = '+' if val >= 0 else ''
            lines.append(f'    {name:20s} -> {sign}{val:,.0f} TL')
        lines.append(f'    diger ozelliklerin toplam katkisi -> {"+" if rest >= 0 else ""}{rest:,.0f} TL')
        lines.append(f'  TOPLAM TAHMIN: {pred:,.0f} TL' + (f' (gercek: {actual:,.0f} TL)' if actual is not None else ''))
        lines.append('')
    return lines


# Madde 4: shap_values + base_value'nun model.predict ile tutarli olup olmadigini dogrulayan
# otomatik test - SHAP toplaminin tahmine ulasmadigi bir rapor kullaniciyi yanlis yonlendirir.
def shap_sum_check(explainer, shap_values, sample, model, tolerance=1.0):
    base = explainer.expected_value
    reconstructed = base + shap_values.sum(axis=1)
    actual_pred = model.predict(sample)
    max_diff = np.max(np.abs(reconstructed - actual_pred))
    passed = max_diff < tolerance
    return passed, max_diff


def main():
    meta = collect_run_metadata()
    artifact = load_model()
    model = artifact['model']

    X_full, y_full = prepare_full_training_data()
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False, encoding='utf-8-sig')
    holdout_df = load_cars1_holdout()
    X_holdout, y_holdout = prepare_external_holdout(X_full, y_full)
    preds = model.predict(X_holdout)

    report = []
    report.append('=== Calistirma metadata (Faz 11 Madde 5) ===')
    for k, v in meta.items():
        report.append(f'{k}: {v}')
    report.append('')

    report.append('=== Dis holdout genel performans (Faz 11 Madde 1) ===')
    report.append(format_metrics('cars1_normalized.csv (hic gorulmemis)', evaluate_metrics(y_holdout.values, preds)))
    report.append('')

    report.append('=== Segment bazli performans: agir_hasarli (Faz 11 Madde 2) ===')
    report.extend(segment_performance(X_holdout, y_holdout, preds, X_full=X_full, y_full=y_full, model=model))
    report.append('')

    report.append('=== Duplicate / cross-source leakage kontrolu (Faz 11 Madde 3) ===')
    report.extend(duplicate_leakage_check(train_df, holdout_df, y_holdout, preds))
    report.append('')

    print('SHAP hesaplaniyor (sample={})...'.format(SHAP_SAMPLE_SIZE))
    sample = X_full.sample(min(SHAP_SAMPLE_SIZE, len(X_full)), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    report.append('=== Global SHAP siralamasi - mean |SHAP| TL (Faz 11 Madde 4) ===')
    report.extend(global_shap_ranking(shap_values, sample))
    report.append('')

    passed, max_diff = shap_sum_check(explainer, shap_values, sample, model)
    report.append('=== SHAP toplam-tutarlilik testi (Faz 11 Madde 4) ===')
    report.append(f'base_value + shap_values.sum() == model.predict() : '
                  f'{"PASS" if passed else "FAIL"} (max fark: {max_diff:.6f} TL)')
    report.append('')

    summary_path = save_summary_plot(shap_values, sample)
    dependence_paths = save_dependence_plots(shap_values, sample)
    report.append('=== Kaydedilen SHAP grafikleri (Faz 11 Madde 4) ===')
    report.append(summary_path)
    report.extend(dependence_paths)
    report.append('')

    report.append('=== Yerel aciklama ornekleri (Faz 11 Madde 4) ===')
    report.extend(local_explanations(explainer, X_holdout, y_holdout, preds, X_full))

    os.makedirs(os.path.dirname(REPORT_TXT_PATH), exist_ok=True)
    with open(REPORT_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f'\nrapor kaydedildi: {REPORT_TXT_PATH}')


if __name__ == '__main__':
    main()
