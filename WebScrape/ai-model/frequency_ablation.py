"""Faz 19: model_frequency / brand_model_frequency / hiyerarsik fiyat
ozellikleri icin ablation testi. Faz 15 (error_taxonomy_analysis.py) model
frekansi ile hata arasinda gucIu bir iliski bulmustu (freq 1-5 -> MAE ~2.9x
freq 500+'a gore) - bu, LightGBM'in kategorik bolme mantiginin (sadece
bolme kazancina bakar) bu FREKANS bilgisini KENDISI turetip turetmedigini
belirsiz birakiyor. Bu script, frekansi ACIKCA sayisal bir ozellik olarak
eklemenin GERCEK ek katki saglayip saglamadigini olcer.

SIZINTI ONLEME (kritik metodolojik kural):
- model_frequency / brand_model_frequency: SADECE egitim (X_full) verisinden
  hesaplanir; dis holdout icin egitim sozlugunden LOOKUP yapilir, gorulmemis
  deger -> 0.
- E grubu (hiyerarsik fiyat medyanlari): egitim SATIRLARINA dogrudan TUM
  egitim verisinden hesaplanan medyan ATANIRSA bu, satirin KENDI fiyatinin
  kendi grubunun medyanina sizmasi (klasik target-leakage) demektir. Bu
  yuzden egitim satirlari icin 5-fold OUT-OF-FOLD hesaplama kullanilir - her
  fold'un medyani DIGER fold'lardan hesaplanir. Dis holdout icin ise fold'suz,
  TUM egitim verisinden hesaplanan medyan guvenlidir (holdout zaten bu
  hesaplamaya hic girmiyor, sizinti riski yok).

Calistirma (ai-model/ calisma dizini olarak): python frequency_ablation.py
"""
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from train import BASELINE_PARAMS, prepare_external_holdout, prepare_full_training_data

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'output', 'frequency_ablation_report.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'data', 'output', 'frequency_ablation_summary.txt')

SEEDS = [42, 123, 7]
RARE_MODEL_FREQ_THRESHOLD = 20  # Faz 15 konvansiyonu: freq<=20 -> "nadir model"
FREQ_BUCKET_EDGES = [0, 5, 20, 100, 500, 10 ** 9]
FREQ_BUCKET_LABELS = ['1-5', '6-20', '21-100', '101-500', '500+']
OOF_N_SPLITS = 5

GROUPS = {
    'A_baseline': {'model_freq': False, 'brand_model_freq': False, 'hier_price': False},
    'B_model_freq': {'model_freq': True, 'brand_model_freq': False, 'hier_price': False},
    'C_brand_model_freq': {'model_freq': False, 'brand_model_freq': True, 'hier_price': False},
    'D_both_freq': {'model_freq': True, 'brand_model_freq': True, 'hier_price': False},
    'E_hierarchical': {'model_freq': True, 'brand_model_freq': True, 'hier_price': True},
}


def compute_freq_dicts(X_train):
    model_freq = X_train['model'].value_counts().to_dict()
    brand_model_freq = X_train.groupby(['marka', 'model'], observed=True).size().to_dict()
    return model_freq, brand_model_freq


def attach_freq_features(X, model_freq, brand_model_freq, add_model_freq, add_brand_model_freq):
    X = X.copy()
    if add_model_freq:
        X['model_frequency'] = X['model'].map(model_freq).fillna(0).astype(float)
    if add_brand_model_freq:
        keys = list(zip(X['marka'], X['model']))
        X['brand_model_frequency'] = np.array([brand_model_freq.get(k, 0) for k in keys], dtype=float)
    return X


def compute_oof_price_stats(X_full, y_full, n_splits=OOF_N_SPLITS, seed=42):
    """Egitim satirlari icin sizinti-siz (out-of-fold) marka/marka-model
    medyan fiyati - her fold'un degeri DIGER fold'lardan hesaplanir."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X_full)
    brand_median = np.full(n, np.nan)
    brand_model_median = np.full(n, np.nan)
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)

    for fold_train_idx, fold_eval_idx in kf.split(df):
        fold_train = df.iloc[fold_train_idx]
        bm = fold_train.groupby('marka', observed=True)['fiyat'].median()
        bmm = fold_train.groupby(['marka', 'model'], observed=True)['fiyat'].median().to_dict()
        overall = fold_train['fiyat'].median()

        fold_eval = df.iloc[fold_eval_idx]
        brand_median[fold_eval_idx] = fold_eval['marka'].map(bm).fillna(overall).values
        keys = list(zip(fold_eval['marka'], fold_eval['model']))
        brand_median_map = bm.to_dict()
        brand_model_median[fold_eval_idx] = [
            bmm.get(k, brand_median_map.get(k[0], overall)) for k in keys
        ]
    return brand_median, brand_model_median


def compute_holdout_price_stats(X_full, y_full, X_holdout):
    """Dis holdout icin FOLD'SUZ, tum egitim verisinden hesaplanir - holdout
    zaten bu hesaplamaya hic dahil olmadigi icin sizinti riski YOK."""
    df = X_full[['marka', 'model']].assign(fiyat=y_full.values)
    bm = df.groupby('marka', observed=True)['fiyat'].median()
    bmm = df.groupby(['marka', 'model'], observed=True)['fiyat'].median().to_dict()
    overall = df['fiyat'].median()
    bm_map = bm.to_dict()

    brand_median = X_holdout['marka'].map(bm).fillna(overall).values
    keys = list(zip(X_holdout['marka'], X_holdout['model']))
    brand_model_median = np.array([bmm.get(k, bm_map.get(k[0], overall)) for k in keys])
    return brand_median, brand_model_median


def regression_stats(y_true, y_pred):
    error = y_pred - y_true
    abs_error = np.abs(error)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        'n': len(y_true), 'mae': abs_error.mean(), 'rmse': np.sqrt((error ** 2).mean()),
        'r2': r2, 'medae': np.median(abs_error), 'bias': error.mean(),
    }


def run_experiment(group_name, config, X_full, y_full, X_holdout, y_holdout, seed, model_freq_train):
    model_freq, brand_model_freq = compute_freq_dicts(X_full)

    X_train_aug = attach_freq_features(X_full, model_freq, brand_model_freq,
                                        config['model_freq'], config['brand_model_freq'])
    X_holdout_aug = attach_freq_features(X_holdout, model_freq, brand_model_freq,
                                          config['model_freq'], config['brand_model_freq'])

    if config['hier_price']:
        brand_med_oof, brand_model_med_oof = compute_oof_price_stats(X_full, y_full, seed=seed)
        X_train_aug = X_train_aug.assign(
            brand_median_price=brand_med_oof, brand_model_median_price=brand_model_med_oof)
        brand_med_holdout, brand_model_med_holdout = compute_holdout_price_stats(X_full, y_full, X_holdout)
        X_holdout_aug = X_holdout_aug.assign(
            brand_median_price=brand_med_holdout, brand_model_median_price=brand_model_med_holdout)

    params = {**BASELINE_PARAMS, 'random_state': seed}
    model = LGBMRegressor(**params)
    t0 = time.perf_counter()
    model.fit(X_train_aug, y_full)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_holdout_aug)
    predict_seconds = time.perf_counter() - t0
    latency_ms_per_row = (predict_seconds / len(X_holdout_aug)) * 1000

    # tek-satir inference latency (production /predict senaryosuna daha yakin)
    single_row = X_holdout_aug.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict(single_row)
    single_row_ms = ((time.perf_counter() - t0) / 200) * 1000

    y_true = y_holdout.values
    overall = regression_stats(y_true, preds)

    hp = X_holdout['motor_gucu'].values
    hp300 = regression_stats(y_true[hp >= 300], preds[hp >= 300]) if (hp >= 300).any() else None

    rare_mask = X_holdout['model'].map(lambda m: model_freq_train.get(m, 0)).values <= RARE_MODEL_FREQ_THRESHOLD
    rare = regression_stats(y_true[rare_mask], preds[rare_mask]) if rare_mask.any() else None

    freq_buckets = {}
    holdout_model_freq = X_holdout['model'].map(lambda m: model_freq_train.get(m, 0)).values
    bucket_idx = pd.cut(holdout_model_freq, FREQ_BUCKET_EDGES, labels=FREQ_BUCKET_LABELS)
    for label in FREQ_BUCKET_LABELS:
        mask = np.asarray(bucket_idx == label)
        if mask.sum() > 0:
            freq_buckets[label] = regression_stats(y_true[mask], preds[mask])

    # yeni ozelliklerin gain-tabanli SHAP-esdeger onemi (pred_contrib - hizli, LightGBM native)
    new_cols = [c for c in ['model_frequency', 'brand_model_frequency',
                             'brand_median_price', 'brand_model_median_price']
                if c in X_train_aug.columns]
    feature_importance = {}
    if new_cols:
        gains = dict(zip(X_train_aug.columns, model.booster_.feature_importance(importance_type='gain')))
        total_gain = sum(gains.values())
        for col in new_cols:
            feature_importance[col] = 100 * gains.get(col, 0) / total_gain if total_gain > 0 else 0.0

    return {
        'group': group_name, 'seed': seed,
        'fit_seconds': fit_seconds, 'latency_ms_per_row_bulk': latency_ms_per_row,
        'latency_ms_single_row': single_row_ms,
        'overall': overall, 'hp300': hp300, 'rare_model': rare,
        'freq_buckets': freq_buckets, 'feature_importance_pct_gain': feature_importance,
    }


def main():
    X_full, y_full = prepare_full_training_data()
    X_holdout, y_holdout = prepare_external_holdout(X_full)
    model_freq_train, _ = compute_freq_dicts(X_full)

    all_results = []
    for group_name, config in GROUPS.items():
        for seed in SEEDS:
            print(f'calistiriliyor: {group_name} seed={seed} ...')
            result = run_experiment(group_name, config, X_full, y_full, X_holdout, y_holdout,
                                     seed, model_freq_train)
            all_results.append(result)

    # --- CSV: satir bazli (group, seed, segment) ---
    rows = []
    for r in all_results:
        segments = [('GENEL', r['overall']), ('300+HP', r['hp300']), ('nadir_model', r['rare_model'])]
        segments += [(f'freq_{label}', stats) for label, stats in r['freq_buckets'].items()]
        for seg_label, stats in segments:
            if stats is None:
                continue
            rows.append({
                'group': r['group'], 'seed': r['seed'], 'segment': seg_label,
                'n': stats['n'], 'mae': stats['mae'], 'rmse': stats['rmse'],
                'r2': stats['r2'], 'medae': stats['medae'], 'bias': stats['bias'],
                'fit_seconds': r['fit_seconds'], 'latency_ms_single_row': r['latency_ms_single_row'],
            })
    report_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    report_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    # --- summary ---
    lines = ['=== Faz 19: model_frequency / brand_model_frequency ablation testi ===']
    lines.append(f'Seed sayisi: {len(SEEDS)} ({SEEDS}), dis holdout: {len(X_holdout):,} kayit')
    lines.append('')

    for group_name in GROUPS:
        lines.append(f'--- {group_name} ---')
        group_results = [r for r in all_results if r['group'] == group_name]
        for seg_key, seg_label in [('overall', 'GENEL'), ('hp300', '300+ HP'), ('rare_model', 'nadir model')]:
            vals = [r[seg_key] for r in group_results if r[seg_key] is not None]
            if not vals:
                continue
            mae_mean = np.mean([v['mae'] for v in vals])
            mae_std = np.std([v['mae'] for v in vals])
            r2_mean = np.mean([v['r2'] for v in vals])
            medae_mean = np.mean([v['medae'] for v in vals])
            n = vals[0]['n']
            lines.append(f'  {seg_label:14s} n={n:>6,} MAE={mae_mean:>10,.0f} (±{mae_std:,.0f})  '
                         f'R2={r2_mean:.4f}  MedAE={medae_mean:>9,.0f}')
        fit_s = np.mean([r['fit_seconds'] for r in group_results])
        lat = np.mean([r['latency_ms_single_row'] for r in group_results])
        lines.append(f'  fit_suresi_ort={fit_s:.2f}s  tek_satir_inference_ort={lat:.3f}ms')
        if group_results[0]['feature_importance_pct_gain']:
            imp_str = ', '.join(f'{k}=%{np.mean([r["feature_importance_pct_gain"][k] for r in group_results]):.1f}'
                                for k in group_results[0]['feature_importance_pct_gain'])
            lines.append(f'  yeni ozellik gain payi: {imp_str}')
        lines.append('')

    lines.append('=== Frekans grubuna gore MAE (grup x seed ortalamasi) ===')
    header = f'{"frekans_grubu":14s}' + ''.join(f'{g:>20s}' for g in GROUPS)
    lines.append(header)
    for label in FREQ_BUCKET_LABELS:
        row = [f'{label:14s}']
        for group_name in GROUPS:
            group_results = [r for r in all_results if r['group'] == group_name]
            vals = [r['freq_buckets'][label]['mae'] for r in group_results if label in r['freq_buckets']]
            row.append(f'{np.mean(vals):>20,.0f}' if vals else f'{"n/a":>20s}')
        lines.append(''.join(row))
    lines.append('')

    # --- karar kriteri ---
    baseline_results = [r for r in all_results if r['group'] == 'A_baseline']
    baseline_mae = np.mean([r['overall']['mae'] for r in baseline_results])
    baseline_rare_mae = np.mean([r['rare_model']['mae'] for r in baseline_results if r['rare_model']])

    lines.append('=== Karar kriteri degerlendirmesi ===')
    for group_name in ['B_model_freq', 'C_brand_model_freq', 'D_both_freq', 'E_hierarchical']:
        group_results = [r for r in all_results if r['group'] == group_name]
        mae_vals = [r['overall']['mae'] for r in group_results]
        rare_vals = [r['rare_model']['mae'] for r in group_results if r['rare_model']]
        mae_mean = np.mean(mae_vals)
        rare_mean = np.mean(rare_vals) if rare_vals else float('nan')
        mae_delta_pct = 100 * (mae_mean - baseline_mae) / baseline_mae
        rare_delta_pct = 100 * (rare_mean - baseline_rare_mae) / baseline_rare_mae if rare_vals else float('nan')
        seed_directions = [1 if r['overall']['mae'] < baseline_mae else -1 for r in group_results]
        consistent = len(set(seed_directions)) == 1
        lines.append(f'{group_name}: GENEL MAE {mae_delta_pct:+.2f}%  nadir_model MAE {rare_delta_pct:+.2f}%  '
                     f'seed_yonu_tutarli={consistent} ({seed_directions})')
    lines.append('')
    lines.append('NOT: "iyilesme" = MAE DUSMESI (negatif % = daha iyi). Karar kriteri: dis holdout GENEL '
                 'MAE anlamli iyilesme + nadir model segmentinde belirgin fayda + normal segmentleri '
                 'kotulestirmiyor + seed yonu tutarli. Tek seferlik R2 dorduncu ondalik artisi YETERLI '
                 'KANIT SAYILMAZ (bkz. gorev talebi).')

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nCSV: {CSV_PATH}')
    print(f'Ozet: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
