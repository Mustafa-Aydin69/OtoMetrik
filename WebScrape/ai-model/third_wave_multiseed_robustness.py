"""Faz 34 - third-wave (27 satir) sentetik agirlik ablation'inin 8-seed
saglamlik testi. Wave30 (18 satir) VE Wave31 (38 satir) HER kolda AYNI
sabit W=0.50 ile korunur - bu deney SADECE Wave34'un (27 satir) EK etkisini
izole eder. Uretim artefaktini DEGISTIRMEZ, hicbir CSV'ye yazmaz.

A = gercek + wave30(w=0.50) + wave31(w=0.50)                    [wave34 YOK]
B = A + wave34(w=0.25)
C = A + wave34(w=0.50)
D = A + wave34(w=1.00)

hierarchical_price: SADECE gercek train'den - UC sentetik dalga da HARIC,
sadece attach_lookup_feature ile SONUCTAN okur.

Calistirma (ai-model/ calisma dizini olarak): python third_wave_multiseed_robustness.py
"""
import json
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
from train import BASELINE_PARAMS, CATEGORICAL_COLS, apply_saved_categories
import hierarchical_price as hp
from hp_support import lookup_support, compute_confidence, build_support_summary

WAVE30_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
WAVE31_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_second_wave_preview.csv')
WAVE34_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_third_wave_preview.csv')
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'production_probe_baseline.json')

SEEDS = [42, 7, 21, 73, 123, 2026, 314, 999]
WAVE3031_WEIGHT = 0.50
PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
WAVE34_GROUPS = [('Audi', 'TTS'), ('Porsche', 'Boxster'), ('Mercedes - Benz', 'GLS'),
                  ('Audi', 'Q4 Sportback'), ('Maserati', 'GranTurismo'), ('Audi', 'Q8 Sportback E-Tron')]
WEIGHT_MODELS = [('B_W025', 0.25), ('C_W050', 0.50), ('D_W100', 1.00)]

TTS_MK2 = {'kaggle-ab-15454', 'kaggle-ab-9403', 'arabam-41457812', 'kaggle-ab-37517',
           'arabam-41743097', 'kaggle-ab-8014', 'kaggle-ab-21325'}
TTS_MK3 = {'arabam-40086630'}
TTS_ANOMALY = {'arabam-40704969'}
BOXSTER_986 = {'arabam-40488741', 'arabam-42139768', 'kaggle-ab-2420', 'kaggle-ab-2392'}
BOXSTER_987_981 = {'kaggle-ab-2402', 'arabam-37244743', 'kaggle-ab-2416', 'kaggle-ab-3847'}
GT_42L = {'kaggle-ab-991', 'kaggle-ab-974', 'arabam-29455568', 'arabam-37773040'}
GT_47L = {'kaggle-ab-984', 'kaggle-ab-975', 'kaggle-ab-981', 'arabam-41561742', 'kaggle-ab-990',
          'arabam-39045347', 'arabam-41697784', 'arabam-42005171', 'arabam-42176084'}
Q4_288 = {'arabam-41702587', 'arabam-41628212', 'arabam-41140561', 'arabam-41475659'}
Q4_213 = {'arabam-42134577'}

PROBE_CASES = [
    ('Audi', 'TTS', 2010, 100000, 'Coupe', 'Kırmızı', 1900.0, 263.0, 'Belirtilmemiş'),
    ('Porsche', 'Boxster', 2007, 150000, 'Roadster', 'Gri', 2750.0, 238.0, 'Belirtilmemiş'),
    ('Mercedes - Benz', 'GLS', 2016, 200000, 'SUV', 'Siyah', 2987.0, 258.0, 'D'),
    ('Audi', 'Q4 Sportback', 2024, 30000, 'SUV', 'Beyaz', np.nan, 288.0, 'e-Tron'),
    ('Maserati', 'GranTurismo', 2011, 80000, 'Coupe', 'Siyah', 4750.0, 438.0, '4.7 S'),
    ('Audi', 'Q8 Sportback E-Tron', 2023, 45000, 'SUV', 'Siyah', np.nan, 408.0, 'S Line'),
]


def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'n': 0, 'MAE': None, 'RMSE': None, 'MAPE%': None, 'R2': None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    return {'n': len(y_true), 'MAE': mae, 'RMSE': rmse, 'MAPE%': mape, 'R2': r2}


def fmt(m):
    if m['n'] == 0:
        return 'n=0'
    r2 = f"{m['R2']:.4f}" if m['R2'] is not None and not (isinstance(m['R2'], float) and np.isnan(m['R2'])) else 'n/a'
    return f"n={m['n']:<6} MAE={m['MAE']:>11,.0f} MAPE={m['MAPE%']:>6.1f}% R2={r2}"


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


def prep_synth(raw, ref_cols):
    cols = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
            'yil', 'kilometre', 'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli']
    X = raw[cols].copy()
    y = raw['fiyat'].reset_index(drop=True)
    X['degisen_sayisi_bilinmiyor'] = 0
    X['boyali_sayisi_bilinmiyor'] = 0
    X['yas'] = (PRICE_REFERENCE_DATE.year - X['yil']).clip(lower=0)
    X['km_yil'] = X['kilometre'] / X['yas'].replace(0, 1)
    X = X.reindex(columns=ref_cols)
    return X, y


def cluster_of(ilan_id, sets_labels):
    for s, label in sets_labels:
        if ilan_id in s:
            return label
    return 'other'


def run_seed(seed, X_full, y_full, wave30_raw, wave31_raw, wave34_raw):
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

    w30_X, w30_y = prep_synth(wave30_raw, X_train.columns)
    w31_X, w31_y = prep_synth(wave31_raw, X_train.columns)
    w34_X, w34_y = prep_synth(wave34_raw, X_train.columns)

    X_train_c, X_test_c, w30_c, w31_c, w34_c = to_cat(X_train, X_test, w30_X, w31_X, w34_X)

    X_train_hp, _ = hp.attach_oof_feature(X_train_c, y_train)
    lookup_real = hp.build_price_lookup(X_train_c, y_train)
    X_test_hp = hp.attach_lookup_feature(X_test_c, lookup_real)
    w30_hp = hp.attach_lookup_feature(w30_c, lookup_real)
    w31_hp = hp.attach_lookup_feature(w31_c, lookup_real)
    w34_hp = hp.attach_lookup_feature(w34_c, lookup_real)

    w_real = pd.Series(1.0, index=X_train_hp.index)
    w_3031 = pd.Series(WAVE3031_WEIGHT, index=range(len(w30_hp) + len(w31_hp)))

    X_a = pd.concat([X_train_hp, w30_hp, w31_hp], ignore_index=True)
    y_a = pd.concat([y_train.reset_index(drop=True), w30_y, w31_y], ignore_index=True)
    w_a = pd.concat([w_real.reset_index(drop=True), w_3031], ignore_index=True)

    model_a = LGBMRegressor(**BASELINE_PARAMS)
    model_a.fit(X_a, y_a, sample_weight=w_a)
    preds = {'A': model_a.predict(X_test_hp)}
    trained_models = {'A': model_a}

    for label, w in WEIGHT_MODELS:
        X_b = pd.concat([X_a, w34_hp], ignore_index=True)
        y_b = pd.concat([y_a, w34_y], ignore_index=True)
        w_b = pd.concat([w_a, pd.Series([w] * len(w34_hp))], ignore_index=True)
        model = LGBMRegressor(**BASELINE_PARAMS)
        model.fit(X_b, y_b, sample_weight=w_b)
        preds[label] = model.predict(X_test_hp)
        trained_models[label] = model

    y_test_arr = y_test.values
    marka_test = X_test['marka'].astype(str).values
    model_test = X_test['model'].astype(str).values
    ids_test = X_test.index.values
    support_counts = X_train.groupby(['marka', 'model'], observed=True).size()
    support = np.array([int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)])

    mask_5m = y_test_arr > 5_000_000
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)
    seg_masks = {
        'overall': np.ones(len(y_test_arr), dtype=bool), '>5M': mask_5m, 'premium': mask_premium,
        'sup<3': support < 3, 'sup3-9': (support >= 3) & (support < 10),
        'sup10-49': (support >= 10) & (support < 50), 'sup50+': support >= 50,
    }
    seed_result = {'seed': seed, 'segments': {}, 'wave34': {}, 'clusters': {}}
    for label in preds:
        seed_result['segments'][label] = {seg: metrics(y_test_arr[m], preds[label][m]) for seg, m in seg_masks.items()}

    for marka, model in WAVE34_GROUPS:
        mask = (marka_test == marka) & (model_test == model)
        n = int(mask.sum())
        entry = {'n_test': n}
        for label in preds:
            entry[f'{label}_total_abs_err'] = float(np.sum(np.abs(preds[label][mask] - y_test_arr[mask]))) if n > 0 else None
        seed_result['wave34'][f'{marka}|{model}'] = entry

    def cluster_report(marka, model, sets_labels):
        mask = (marka_test == marka) & (model_test == model)
        if mask.sum() == 0:
            return None
        idxs = np.where(mask)[0]
        ids_here = ids_test[idxs]
        labels_here = np.array([cluster_of(i, sets_labels) for i in ids_here])
        out = {}
        for lbl in set(labels_here):
            sub_idx = idxs[labels_here == lbl]
            out[lbl] = {'n': len(sub_idx)}
            for label in preds:
                out[lbl][f'{label}_abs_err'] = float(np.sum(np.abs(preds[label][sub_idx] - y_test_arr[sub_idx])))
        return out

    seed_result['clusters']['Audi|TTS'] = cluster_report('Audi', 'TTS', [(TTS_MK2, 'mk2'), (TTS_MK3, 'mk3'), (TTS_ANOMALY, 'anomaly')])
    seed_result['clusters']['Porsche|Boxster'] = cluster_report('Porsche', 'Boxster', [(BOXSTER_986, '986'), (BOXSTER_987_981, '987_981')])
    seed_result['clusters']['Maserati|GranTurismo'] = cluster_report('Maserati', 'GranTurismo', [(GT_42L, '4.2L'), (GT_47L, '4.7L')])
    seed_result['clusters']['Audi|Q4 Sportback'] = cluster_report('Audi', 'Q4 Sportback', [(Q4_288, '288hp'), (Q4_213, '213hp')])

    return seed_result, trained_models, lookup_real, X_train_hp


def main():
    print('=== gercek veri (BIR KEZ) ===')
    clean = load_clean_train_dataset()
    y_full = clean['fiyat']
    X_full = clean.drop(columns=['fiyat', 'ilan_id'])
    X_full.index = clean['ilan_id'].values

    print('=== sabit sentetikler okunuyor (Wave30=18, Wave31=38, Wave34=27) ===')
    wave30_raw = pd.read_csv(WAVE30_PATH)
    wave31_raw = pd.read_csv(WAVE31_PATH)
    wave34_raw = pd.read_csv(WAVE34_PATH)
    assert len(wave30_raw) == 18 and len(wave31_raw) == 38 and len(wave34_raw) == 27
    print(f'wave30={len(wave30_raw)}, wave31={len(wave31_raw)}, wave34={len(wave34_raw)}')

    all_results = []
    last_trained = None
    for seed in SEEDS:
        print(f'\n########## SEED {seed} ##########')
        r, trained_models, lookup_real, X_train_hp = run_seed(seed, X_full, y_full, wave30_raw, wave31_raw, wave34_raw)
        all_results.append(r)
        last_trained = (trained_models, lookup_real, X_train_hp)
        for label in ['A', 'B_W025', 'C_W050', 'D_W100']:
            print(f'--- {label} ---')
            for seg in ['overall', '>5M', 'premium', 'sup<3', 'sup3-9', 'sup10-49', 'sup50+']:
                print(f'  {seg:<10} {fmt(r["segments"][label][seg])}')
        print('  wave34:')
        for key, entry in r['wave34'].items():
            print(f'    {key}: n_test={entry["n_test"]} A={entry["A_total_abs_err"]} '
                  f'B={entry["B_W025_total_abs_err"]} C={entry["C_W050_total_abs_err"]} D={entry["D_W100_total_abs_err"]}')
        print('  clusters:')
        for key, val in r['clusters'].items():
            if val:
                print(f'    {key}: {val}')

    print('\n\n=== AGGREGATE (8 seed) ===')

    def agg(label, seg, field):
        vals = [r['segments'][label][seg][field] for r in all_results if r['segments'][label][seg]['n'] > 0]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    labels = ['A', 'B_W025', 'C_W050', 'D_W100']
    print('\nMetric | ' + ' | '.join(f'{l} mean±std' for l in labels))
    for seg, field, name in [('overall', 'R2', 'overall R2'), ('overall', 'MAE', 'overall MAE'),
                              ('>5M', 'R2', '>5M R2'), ('>5M', 'MAE', '>5M MAE'),
                              ('premium', 'MAE', 'premium MAE'), ('premium', 'R2', 'premium R2')]:
        parts = []
        for l in labels:
            m, s = agg(l, seg, field)
            parts.append(f'{m:,.4f}±{s:,.4f}' if m is not None else 'n/a')
        print(f'{name:<15} ' + ' | '.join(parts))

    print('\nthird-wave 6 grup toplam absolute error (sadece n_test>0 seedler):')
    totals = {l: [] for l in labels}
    for r in all_results:
        n_any = sum(e['n_test'] for e in r['wave34'].values())
        if n_any == 0:
            continue
        for l in labels:
            key = 'A_total_abs_err' if l == 'A' else f'{l}_total_abs_err'
            tot = sum(e[key] for e in r['wave34'].values() if e[key] is not None)
            totals[l].append(tot)
    for l in labels:
        v = totals[l]
        if v:
            print(f'  {l}: mean={np.mean(v):,.0f} std={np.std(v):,.0f} (n_seeds={len(v)})')

    print('\n=== KAC SEEDDE IYI ===')
    for l in ['B_W025', 'C_W050', 'D_W100']:
        overall_ok = sum(1 for r in all_results if r['segments'][l]['overall']['MAE'] <= r['segments']['A']['overall']['MAE'] * 1.01)
        m5_better = sum(1 for r in all_results if r['segments'][l]['>5M']['n'] > 0 and r['segments'][l]['>5M']['MAE'] < r['segments']['A']['>5M']['MAE'])
        prem_better = sum(1 for r in all_results if r['segments'][l]['premium']['n'] > 0 and r['segments'][l]['premium']['MAE'] < r['segments']['A']['premium']['MAE'])
        print(f'{l}: overall_not_worse(<=101%)={overall_ok}/8  >5M_better={m5_better}/8  premium_better={prem_better}/8')

    n_seeds_w34 = 0
    w34_better = {l: 0 for l in ['B_W025', 'C_W050', 'D_W100']}
    for r in all_results:
        n_any = sum(e['n_test'] for e in r['wave34'].values())
        if n_any == 0:
            continue
        n_seeds_w34 += 1
        a_tot = sum(e['A_total_abs_err'] for e in r['wave34'].values() if e['A_total_abs_err'] is not None)
        for l in ['B_W025', 'C_W050', 'D_W100']:
            tot = sum(e[f'{l}_total_abs_err'] for e in r['wave34'].values() if e[f'{l}_total_abs_err'] is not None)
            if tot < a_tot:
                w34_better[l] += 1
    for l in ['B_W025', 'C_W050', 'D_W100']:
        print(f'third-wave toplam hata {l} iyi: {w34_better[l]}/{n_seeds_w34}')

    print('\n=== PER-MODEL: kac seedde her weight baseline dan iyi ===')
    for marka, model in WAVE34_GROUPS:
        key = f'{marka}|{model}'
        counts = {l: 0 for l in ['B_W025', 'C_W050', 'D_W100']}
        n_seeds_present = 0
        for r in all_results:
            e = r['wave34'][key]
            if e['n_test'] == 0:
                continue
            n_seeds_present += 1
            for l in counts:
                if e[f'{l}_total_abs_err'] < e['A_total_abs_err']:
                    counts[l] += 1
        print(f'{key}: n_seeds_present={n_seeds_present}  ' + '  '.join(f'{l}={v}/{n_seeds_present}' for l, v in counts.items()))

    print('\n=== CLUSTER OZETI (8 seed) ===')
    for r in all_results:
        for key, val in r['clusters'].items():
            if val:
                print(f'seed {r["seed"]} {key}: {val}')

    print('\n=== PROBE (sabit girdi, son seed modelleri) ===')
    trained_models, lookup_real, X_train_hp = last_trained
    reference_year = PRICE_REFERENCE_DATE.year
    for marka, model, yil, km, kasa, renk, mh, mg, paket in PROBE_CASES:
        yas = max(reference_year - yil, 0)
        km_yil = km / (yas if yas > 0 else 1)
        hp_val, hp_src, hp_n = hp.lookup_price(marka, model, yas, lookup_real)
        row = pd.DataFrame([{
            'marka': marka, 'model': model, 'paket': paket, 'kasa_turu': kasa, 'renk': renk,
            'motor_hacmi': mh, 'motor_gucu': mg, 'yil': yil, 'kilometre': km, 'yakit_turu': 'Benzin',
            'vites': 'Otomatik', 'degisen_sayisi': 0, 'boyali_sayisi': 0, 'agir_hasarli': 0,
            'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0, 'yas': yas, 'km_yil': km_yil,
        }])
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype('category').cat.set_categories(X_train_hp[c].cat.categories)
        row = row.reindex(columns=X_train_hp.columns)
        row[hp.FEATURE_COLUMN] = hp_val
        print(f'\n{marka} {model} ({yil}, {km:,}km): hp_reference={hp_val:,.0f} hp_source={hp_src} hp_support(fold)={hp_n}')
        for label, m in trained_models.items():
            pred = float(m.predict(row)[0])
            pct_err = 100 * abs(pred - hp_val) / hp_val
            print(f'  {label:<10} pred={pred:,.0f}  pct_err_vs_hp_ref={pct_err:.1f}%')

    print('\n=== FAZ32 REGRESSION FIXTURE KONTROLU (son seed modelleriyle, artifact DEGISMEDI) ===')
    if not os.path.exists(FIXTURE_PATH):
        print('  fixture yok, atlaniyor')
    else:
        with open(FIXTURE_PATH, encoding='utf-8') as f:
            fixture = json.load(f)
        for entry in fixture['main_probes']:
            marka, model = entry['marka'], entry['model']
            payload = entry['input']
            yas = max(reference_year - payload['year'], 0)
            km_yil = payload['mileage'] / (yas if yas > 0 else 1)
            hp_val, hp_src, hp_n = hp.lookup_price(marka, model, yas, lookup_real)
            row = pd.DataFrame([{
                'marka': marka, 'model': model, 'paket': payload['trim'] or 'Belirtilmemiş',
                'kasa_turu': payload['bodyType'], 'renk': payload['color'],
                'motor_hacmi': payload['engineDisplacement'], 'motor_gucu': payload['enginePower'],
                'yil': payload['year'], 'kilometre': payload['mileage'], 'yakit_turu': payload['fuelType'],
                'vites': payload['transmission'], 'degisen_sayisi': payload['replacedPartsCount'],
                'boyali_sayisi': payload['paintedPartsCount'], 'agir_hasarli': int(payload['heavyDamage']),
                'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0, 'yas': yas, 'km_yil': km_yil,
            }])
            for c in CATEGORICAL_COLS:
                row[c] = row[c].astype('category').cat.set_categories(X_train_hp[c].cat.categories)
            row = row.reindex(columns=X_train_hp.columns)
            row[hp.FEATURE_COLUMN] = hp_val
            accepted = entry['accepted_prediction']
            tol = entry['allowed_prediction_delta_pct'] / 100.0
            lo, hi = accepted * (1 - tol), accepted * (1 + tol)
            statuses = []
            for label, m in trained_models.items():
                pred = float(m.predict(row)[0])
                ok = lo <= pred <= hi
                statuses.append(f'{label}={"PASS" if ok else "FAIL(" + format(pred, ",.0f") + ")"}')
            print(f'  {marka}|{model} accepted={accepted:,.0f} band=[{lo:,.0f},{hi:,.0f}]: ' + ' '.join(statuses))


if __name__ == '__main__':
    main()
