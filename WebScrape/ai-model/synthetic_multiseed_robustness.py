"""Faz 30 pilot v3 - coklu-seed saglamlik testi. W=0.50 sonucunun TEK bir
random_state'e bagli olup olmadigini 8 farkli train/test split seed'iyle test
eder. Sentetik veri SABIT (data/output/synthetic_pilot.csv, 18 satir, YENIDEN
URETILMEZ). Uretim artefaktini DEGISTIRMEZ, train_dataset.csv'ye DOKUNMAZ.

Her seed icin: gercek veri o seed'le train/test'e ayrilir (sentetik hicbir
zaman test'e girmez), hierarchical_price SADECE gercek train'den kurulur,
3 model (A=baseline, C=W0.50, D=W1.00) egitilir - AYNI hiperparametreler
(train.BASELINE_PARAMS, kendi ic random_state'i SABIT 42) her seed'de.

Calistirma (ai-model/ calisma dizini olarak): python synthetic_multiseed_robustness.py
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import PRICE_REFERENCE_DATE, load_clean_train_dataset
from train import BASELINE_PARAMS, CATEGORICAL_COLS
import hierarchical_price as hp

SYNTHETIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
SEEDS = [42, 7, 21, 73, 123, 2026, 314, 999]
PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
PILOT_GROUPS = [('Ferrari', '458'), ('Lamborghini', 'Huracan'), ('Rolls-Royce', 'Ghost')]
WEIGHT_MODELS = [('C_W050', 0.50), ('D_W100', 1.00)]


def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'n': 0, 'MAE': None, 'RMSE': None, 'MAPE%': None, 'R2': None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {'n': len(y_true), 'MAE': mae, 'RMSE': rmse, 'MAPE%': mape, 'R2': r2}


def to_cat(X_train, *others):
    X_train = X_train.copy()
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
    result = [X_train]
    for X in others:
        X = X.copy()
        for col in CATEGORICAL_COLS:
            X[col] = X[col].astype('category').cat.set_categories(X_train[col].cat.categories)
        result.append(X)
    return result


def run_seed(seed, X_full, y_full, synth_raw):
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)
    synth_X = synth_raw.copy()
    synth_X['degisen_sayisi_bilinmiyor'] = 0
    synth_X['boyali_sayisi_bilinmiyor'] = 0
    synth_X['yas'] = (PRICE_REFERENCE_DATE.year - synth_X['yil']).clip(lower=0)
    synth_X['km_yil'] = synth_X['kilometre'] / synth_X['yas'].replace(0, 1)
    synth_X = synth_X.reindex(columns=X_train.columns)
    synth_y = pd.Series(synth_raw['fiyat'].values)

    X_train_c, X_test_c, synth_X_c = to_cat(X_train, X_test, synth_X)

    X_train_hp, _ = hp.attach_oof_feature(X_train_c, y_train)
    lookup_real = hp.build_price_lookup(X_train_c, y_train)
    X_test_hp = hp.attach_lookup_feature(X_test_c, lookup_real)
    synth_X_hp = hp.attach_lookup_feature(synth_X_c, lookup_real)

    w_real = pd.Series(1.0, index=X_train_hp.index)
    model_a = LGBMRegressor(**BASELINE_PARAMS)
    model_a.fit(X_train_hp, y_train, sample_weight=w_real)

    preds = {'A': model_a.predict(X_test_hp)}
    for label, w in WEIGHT_MODELS:
        X_comb = pd.concat([X_train_hp, synth_X_hp], ignore_index=True)
        y_comb = pd.concat([y_train.reset_index(drop=True), synth_y], ignore_index=True)
        w_comb = pd.concat([w_real.reset_index(drop=True), pd.Series([w] * len(synth_y))], ignore_index=True)
        model = LGBMRegressor(**BASELINE_PARAMS)
        model.fit(X_comb, y_comb, sample_weight=w_comb)
        preds[label] = model.predict(X_test_hp)

    y_test_arr = y_test.values
    marka_test = X_test['marka'].astype(str).values
    model_test = X_test['model'].astype(str).values
    support_counts = X_train.groupby(['marka', 'model'], observed=True).size()
    support = np.array([int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)])

    mask_5m = y_test_arr > 5_000_000
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)
    seg_masks = {
        'overall': np.ones(len(y_test_arr), dtype=bool),
        '>5M': mask_5m,
        'premium': mask_premium,
        'sup<3': support < 3,
        'sup3-9': (support >= 3) & (support < 10),
        'sup10-49': (support >= 10) & (support < 50),
        'sup50+': support >= 50,
    }

    seed_result = {'seed': seed, 'segments': {}, 'pilot': {}}
    for label in preds:
        seed_result['segments'][label] = {seg: metrics(y_test_arr[m], preds[label][m]) for seg, m in seg_masks.items()}

    for marka, model in PILOT_GROUPS:
        mask = (marka_test == marka) & (model_test == model)
        n = int(mask.sum())
        entry = {'n_test': n}
        for label in preds:
            entry[f'{label}_total_abs_err'] = float(np.sum(np.abs(preds[label][mask] - y_test_arr[mask]))) if n > 0 else None
        seed_result['pilot'][f'{marka}|{model}'] = entry

    return seed_result


def fmt_metrics(m):
    if m['n'] == 0:
        return 'n=0'
    r2 = f"{m['R2']:.4f}" if m['R2'] is not None and not (isinstance(m['R2'], float) and np.isnan(m['R2'])) else 'n/a'
    return f"n={m['n']:<6} MAE={m['MAE']:>11,.0f} MAPE={m['MAPE%']:>6.1f}% R2={r2}"


def main():
    print('=== gercek veri hazirlaniyor (BIR KEZ) ===')
    clean = load_clean_train_dataset()
    y_full = clean['fiyat']
    X_full = clean.drop(columns=['fiyat', 'ilan_id'])

    print('=== sabit sentetik pilot okunuyor (SADECE OKUMA, YENIDEN URETILMEZ) ===')
    synth_cols = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
                  'yil', 'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi',
                  'agir_hasarli', 'fiyat']
    synth = pd.read_csv(SYNTHETIC_PATH)
    print(f'{len(synth)} sentetik satir (sabit)')
    synth_raw = synth[synth_cols].copy()

    all_results = []
    for seed in SEEDS:
        print(f'\n########## SEED {seed} ##########')
        r = run_seed(seed, X_full, y_full, synth_raw)
        all_results.append(r)
        for label in ['A', 'C_W050', 'D_W100']:
            print(f'--- {label} ---')
            for seg in ['overall', '>5M', 'premium', 'sup<3', 'sup3-9', 'sup10-49', 'sup50+']:
                print(f'  {seg:<10} {fmt_metrics(r["segments"][label][seg])}')
        print('  pilot:')
        for key, entry in r['pilot'].items():
            print(f'    {key}: n_test={entry["n_test"]}  A={entry["A_total_abs_err"]}  '
                  f'C={entry["C_W050_total_abs_err"]}  D={entry["D_W100_total_abs_err"]}')

    print('\n\n=== AGGREGATE (8 seed) ===')

    def agg(label, seg, field):
        vals = [r['segments'][label][seg][field] for r in all_results if r['segments'][label][seg]['n'] > 0]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    print('\nMetric | Baseline mean±std | W=0.50 mean±std | W=1.00 mean±std')
    for seg, field, name in [('overall', 'R2', 'overall R2'), ('overall', 'MAE', 'overall MAE'),
                              ('>5M', 'R2', '>5M R2'), ('>5M', 'MAE', '>5M MAE'),
                              ('premium', 'MAE', 'premium MAE'), ('premium', 'MAPE%', 'premium MAPE%'),
                              ('premium', 'R2', 'premium R2')]:
        a_m, a_s = agg('A', seg, field)
        c_m, c_s = agg('C_W050', seg, field)
        d_m, d_s = agg('D_W100', seg, field)
        print(f'{name:<15} {a_m:,.4f}±{a_s:,.4f} | {c_m:,.4f}±{c_s:,.4f} | {d_m:,.4f}±{d_s:,.4f}')

    print('\npilot-total absolute error (3 grubun toplami, sadece n_test>0 seedler):')
    pilot_totals = {'A': [], 'C_W050': [], 'D_W100': []}
    for r in all_results:
        for label in pilot_totals:
            tot = sum(e[f'{label}_total_abs_err'] for e in r['pilot'].values() if e[f'{label}_total_abs_err'] is not None)
            n_any = sum(e['n_test'] for e in r['pilot'].values())
            if n_any > 0:
                pilot_totals[label].append(tot)
    for label in pilot_totals:
        v = pilot_totals[label]
        print(f'  {label}: mean={np.mean(v):,.0f} std={np.std(v):,.0f} (n_seeds_with_data={len(v)})')

    print('\n=== KARSILASTIRMA SAYILARI ===')
    c_better = sum(1 for r in all_results if r['segments']['C_W050']['overall']['MAE'] < r['segments']['A']['overall']['MAE'])
    d_better = sum(1 for r in all_results if r['segments']['D_W100']['overall']['MAE'] < r['segments']['A']['overall']['MAE'])
    print(f'W=0.50 kac seedde overall MAE baseline dan iyi: {c_better}/{len(SEEDS)}')
    print(f'W=1.00 kac seedde overall MAE baseline dan iyi: {d_better}/{len(SEEDS)}')

    for marka, model in PILOT_GROUPS:
        key = f'{marka}|{model}'
        n_seeds_present = sum(1 for r in all_results if r['pilot'][key]['n_test'] > 0)
        print(f'{marka} {model}: kac split te testte yer aldi: {n_seeds_present}/{len(SEEDS)}')

    print('\n>5M segment - kac seedde W=0.50/W=1.00 baseline dan iyi (MAE):')
    c_5m_better = sum(1 for r in all_results if r['segments']['C_W050']['>5M']['n'] > 0 and
                       r['segments']['C_W050']['>5M']['MAE'] < r['segments']['A']['>5M']['MAE'])
    d_5m_better = sum(1 for r in all_results if r['segments']['D_W100']['>5M']['n'] > 0 and
                       r['segments']['D_W100']['>5M']['MAE'] < r['segments']['A']['>5M']['MAE'])
    print(f'  W=0.50: {c_5m_better}/{len(SEEDS)}')
    print(f'  W=1.00: {d_5m_better}/{len(SEEDS)}')

    print('\npilot toplam hata - kac seedde W=0.50/W=1.00 baseline dan iyi:')
    c_pilot_better = 0
    d_pilot_better = 0
    n_seeds_pilot_data = 0
    for r in all_results:
        a_tot = sum(e['A_total_abs_err'] for e in r['pilot'].values() if e['A_total_abs_err'] is not None)
        c_tot = sum(e['C_W050_total_abs_err'] for e in r['pilot'].values() if e['C_W050_total_abs_err'] is not None)
        d_tot = sum(e['D_W100_total_abs_err'] for e in r['pilot'].values() if e['D_W100_total_abs_err'] is not None)
        n_any = sum(e['n_test'] for e in r['pilot'].values())
        if n_any == 0:
            continue
        n_seeds_pilot_data += 1
        if c_tot < a_tot:
            c_pilot_better += 1
        if d_tot < a_tot:
            d_pilot_better += 1
    print(f'  W=0.50: {c_pilot_better}/{n_seeds_pilot_data} (veri iceren seed sayisi)')
    print(f'  W=1.00: {d_pilot_better}/{n_seeds_pilot_data}')


if __name__ == '__main__':
    main()
