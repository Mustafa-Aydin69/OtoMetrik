"""Faz 31 - second-wave (38 satir) sentetik agirlik ablation'inin 8-seed
saglamlik testi. Faz30 pilot (18 satir, Ferrari 458/Huracan/Ghost, W=0.50
SABIT) HER kolda aynen korunur - bu deney SADECE Faz31'in 38 satirlik EK
etkisini izole eder. Uretim artefaktini DEGISTIRMEZ, hicbir CSV'ye yazmaz.

A = gercek + Faz30 pilot (w=0.50)
B = A + Faz31 second-wave (w=0.25)
C = A + Faz31 second-wave (w=0.50)
D = A + Faz31 second-wave (w=1.00)

hierarchical_price: SADECE gercek train'den (Faz30 VE Faz31 sentetikleri
HARIC) - her seed'de yeniden kurulur, iki sentetik dalga da SADECE
attach_lookup_feature ile SONUCTAN okur.

Calistirma (ai-model/ calisma dizini olarak): python second_wave_multiseed_robustness.py
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
from hp_support import lookup_support, compute_confidence, build_support_summary

PILOT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
WAVE2_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_second_wave_preview.csv')
SEEDS = [42, 7, 21, 73, 123, 2026, 314, 999]
PILOT_WEIGHT = 0.50
PREMIUM_BRANDS = ['Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce']
WAVE2_GROUPS = [('Dodge', 'Ram'), ('Rolls-Royce', 'Wraith'), ('Lexus', 'LS'),
                ('Aston Martin', 'Vantage'), ('Bentley', 'Flying Spur'),
                ('Mercedes - Benz', 'Maybach S'), ('Mercedes - Benz', 'V Serisi')]
WEIGHT_MODELS = [('B_W025', 0.25), ('C_W050', 0.50), ('D_W100', 1.00)]

# ozel alt-grup ayrimi icin gercek ilan_id kumeleri (Faz31'de tanimlanan nesil kumeleri)
DODGE_RAM_LEGACY = {'arabam-40745482'}
LEXUS_LS_LEGACY = {'arabam-41392003'}
FLYING_SPUR_GEN2 = {'kaggle-ab-272', 'kaggle-ab-271', 'arabam-42433249', 'arabam-39563948'}
FLYING_SPUR_GEN3 = {'arabam-39320430', 'kaggle-ar-30608246', 'arabam-40722555'}

PROBE_CASES = [
    ('Dodge', 'Ram', 2023, 15000, 'SUV', 'Siyah', 2750.5, 413.0),
    ('Rolls-Royce', 'Wraith', 2015, 50000, 'Coupe', 'Siyah', 6592.0, 632.0),
    ('Lexus', 'LS', 2021, 50000, 'Sedan', 'Siyah', 3250.0, 363.0),
    ('Aston Martin', 'Vantage', 2012, 50000, 'Coupe', 'Gri', 4750.0, 438.0),
    ('Bentley', 'Flying Spur', 2021, 42000, 'Sedan', 'Siyah', 3750.5, 538.0),
    ('Mercedes - Benz', 'Maybach S', 2018, 120000, 'Sedan', 'Siyah', 3750.0, 463.0),
    ('Mercedes - Benz', 'V Serisi', 2021, 80000, 'Camlı Van', 'Füme', 1950.0, 237.0),
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


def run_seed(seed, X_full, y_full, pilot_raw, wave2_raw):
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

    pilot_X, pilot_y = prep_synth(pilot_raw, X_train.columns)
    wave2_X, wave2_y = prep_synth(wave2_raw, X_train.columns)

    X_train_c, X_test_c, pilot_X_c, wave2_X_c = to_cat(X_train, X_test, pilot_X, wave2_X)

    X_train_hp, _ = hp.attach_oof_feature(X_train_c, y_train)
    lookup_real = hp.build_price_lookup(X_train_c, y_train)
    X_test_hp = hp.attach_lookup_feature(X_test_c, lookup_real)
    pilot_hp = hp.attach_lookup_feature(pilot_X_c, lookup_real)
    wave2_hp = hp.attach_lookup_feature(wave2_X_c, lookup_real)

    w_real = pd.Series(1.0, index=X_train_hp.index)
    w_pilot = pd.Series(PILOT_WEIGHT, index=range(len(pilot_hp)))

    X_a = pd.concat([X_train_hp, pilot_hp], ignore_index=True)
    y_a = pd.concat([y_train.reset_index(drop=True), pilot_y], ignore_index=True)
    w_a = pd.concat([w_real.reset_index(drop=True), w_pilot], ignore_index=True)

    model_a = LGBMRegressor(**BASELINE_PARAMS)
    model_a.fit(X_a, y_a, sample_weight=w_a)
    preds = {'A': model_a.predict(X_test_hp)}

    for label, w in WEIGHT_MODELS:
        X_b = pd.concat([X_a, wave2_hp], ignore_index=True)
        y_b = pd.concat([y_a, wave2_y], ignore_index=True)
        w_b = pd.concat([w_a, pd.Series([w] * len(wave2_hp))], ignore_index=True)
        model = LGBMRegressor(**BASELINE_PARAMS)
        model.fit(X_b, y_b, sample_weight=w_b)
        preds[label] = model.predict(X_test_hp)

    y_test_arr = y_test.values
    marka_test = X_test['marka'].astype(str).values
    model_test = X_test['model'].astype(str).values
    ilan_test = X_test.index  # orijinal df index kaybolmadi (train_test_split index korur)
    support_counts = X_train.groupby(['marka', 'model'], observed=True).size()
    support = np.array([int(support_counts.get((m, mo), 0)) for m, mo in zip(marka_test, model_test)])

    mask_5m = y_test_arr > 5_000_000
    mask_premium = np.isin(marka_test, PREMIUM_BRANDS)
    seg_masks = {
        'overall': np.ones(len(y_test_arr), dtype=bool), '>5M': mask_5m, 'premium': mask_premium,
        'sup<3': support < 3, 'sup3-9': (support >= 3) & (support < 10),
        'sup10-49': (support >= 10) & (support < 50), 'sup50+': support >= 50,
    }
    seed_result = {'seed': seed, 'segments': {}, 'wave2': {}}
    for label in preds:
        seed_result['segments'][label] = {seg: metrics(y_test_arr[m], preds[label][m]) for seg, m in seg_masks.items()}

    for marka, model in WAVE2_GROUPS:
        mask = (marka_test == marka) & (model_test == model)
        n = int(mask.sum())
        entry = {'n_test': n}
        for label in preds:
            entry[f'{label}_total_abs_err'] = float(np.sum(np.abs(preds[label][mask] - y_test_arr[mask]))) if n > 0 else None
        seed_result['wave2'][f'{marka}|{model}'] = entry

    return seed_result, X_test, y_test, preds


def special_subgroup_report(marka, model, legacy_ids, X_test_full_id, y_test_arr, preds, marka_test, model_test):
    mask_group = (marka_test == marka) & (model_test == model)
    if mask_group.sum() == 0:
        return None
    idxs = np.where(mask_group)[0]
    ids_here = X_test_full_id[idxs]
    is_legacy = np.array([i in legacy_ids for i in ids_here])
    out = {'legacy_n': int(is_legacy.sum()), 'modern_n': int((~is_legacy).sum())}
    for label in preds:
        p = preds[label][idxs]
        a = y_test_arr[idxs]
        if is_legacy.sum() > 0:
            out[f'{label}_legacy_abs_err'] = float(np.sum(np.abs(p[is_legacy] - a[is_legacy])))
        if (~is_legacy).sum() > 0:
            out[f'{label}_modern_abs_err'] = float(np.sum(np.abs(p[~is_legacy] - a[~is_legacy])))
    return out


def main():
    print('=== gercek veri (BIR KEZ) ===')
    clean = load_clean_train_dataset()
    y_full = clean['fiyat']
    X_full = clean.drop(columns=['fiyat', 'ilan_id'])
    X_full.index = clean['ilan_id'].values  # ilan_id'yi index yap - test satirlarini izlemek icin

    print('=== sabit sentetikler okunuyor (Faz30 pilot=18, Faz31 second-wave=38) ===')
    pilot_raw = pd.read_csv(PILOT_PATH)
    wave2_raw = pd.read_csv(WAVE2_PATH)
    assert len(pilot_raw) == 18 and len(wave2_raw) == 38
    print(f'pilot: {len(pilot_raw)}, second-wave: {len(wave2_raw)}')

    all_results = []
    all_special = []
    all_probes = []
    for seed in SEEDS:
        print(f'\n########## SEED {seed} ##########')
        r, X_test, y_test, preds = run_seed(seed, X_full, y_full, pilot_raw, wave2_raw)
        all_results.append(r)
        for label in ['A', 'B_W025', 'C_W050', 'D_W100']:
            print(f'--- {label} ---')
            for seg in ['overall', '>5M', 'premium', 'sup<3', 'sup3-9', 'sup10-49', 'sup50+']:
                print(f'  {seg:<10} {fmt(r["segments"][label][seg])}')
        print('  second-wave:')
        for key, entry in r['wave2'].items():
            print(f'    {key}: n_test={entry["n_test"]} A={entry["A_total_abs_err"]} '
                  f'B={entry["B_W025_total_abs_err"]} C={entry["C_W050_total_abs_err"]} D={entry["D_W100_total_abs_err"]}')

        marka_test = X_test['marka'].astype(str).values
        model_test = X_test['model'].astype(str).values
        y_test_arr = y_test.values
        ids_here = X_test.index.values

        special = {'seed': seed}
        for marka, model, legacy_ids in [('Dodge', 'Ram', DODGE_RAM_LEGACY), ('Lexus', 'LS', LEXUS_LS_LEGACY)]:
            res = special_subgroup_report(marka, model, legacy_ids, ids_here, y_test_arr, preds, marka_test, model_test)
            if res:
                special[f'{marka}|{model}'] = res
                print(f'  [ozel] {marka} {model}: legacy_n={res["legacy_n"]} modern_n={res["modern_n"]}')
        # Flying Spur: iki nesil de "modern" degil, gen2 vs gen3 ayri raporla
        mask_fs = (marka_test == 'Bentley') & (model_test == 'Flying Spur')
        if mask_fs.sum() > 0:
            idxs = np.where(mask_fs)[0]
            ids_fs = ids_here[idxs]
            gen2_mask = np.array([i in FLYING_SPUR_GEN2 for i in ids_fs])
            gen3_mask = np.array([i in FLYING_SPUR_GEN3 for i in ids_fs])
            fs_entry = {'gen2_n': int(gen2_mask.sum()), 'gen3_n': int(gen3_mask.sum()),
                        'other_n': int((~gen2_mask & ~gen3_mask).sum())}
            for label in preds:
                p, a = preds[label][idxs], y_test_arr[idxs]
                if gen2_mask.sum() > 0:
                    fs_entry[f'{label}_gen2_abs_err'] = float(np.sum(np.abs(p[gen2_mask] - a[gen2_mask])))
                if gen3_mask.sum() > 0:
                    fs_entry[f'{label}_gen3_abs_err'] = float(np.sum(np.abs(p[gen3_mask] - a[gen3_mask])))
            special['Bentley|Flying Spur'] = fs_entry
            print(f'  [ozel] Bentley Flying Spur: gen2_n={fs_entry["gen2_n"]} gen3_n={fs_entry["gen3_n"]}')
        all_special.append(special)

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

    print('\nsecond-wave 7 grup toplam absolute error (sadece n_test>0 seedler):')
    totals = {l: [] for l in labels}
    for r in all_results:
        n_any = sum(e['n_test'] for e in r['wave2'].values())
        if n_any == 0:
            continue
        for l in labels:
            key = 'A_total_abs_err' if l == 'A' else f'{l}_total_abs_err'
            tot = sum(e[key] for e in r['wave2'].values() if e[key] is not None)
            totals[l].append(tot)
    for l in labels:
        v = totals[l]
        print(f'  {l}: mean={np.mean(v):,.0f} std={np.std(v):,.0f} (n_seeds={len(v)})')

    print('\n=== KAC SEEDDE IYI (baseline A ile karsilastirma) ===')
    for l in ['B_W025', 'C_W050', 'D_W100']:
        overall_better = sum(1 for r in all_results if r['segments'][l]['overall']['MAE'] < r['segments']['A']['overall']['MAE'])
        m5_better = sum(1 for r in all_results if r['segments'][l]['>5M']['n'] > 0 and r['segments'][l]['>5M']['MAE'] < r['segments']['A']['>5M']['MAE'])
        prem_better = sum(1 for r in all_results if r['segments'][l]['premium']['n'] > 0 and r['segments'][l]['premium']['MAE'] < r['segments']['A']['premium']['MAE'])
        print(f'{l}: overall={overall_better}/8  >5M={m5_better}/8  premium_MAE={prem_better}/8')

    n_seeds_wave2 = 0
    wave2_better = {l: 0 for l in ['B_W025', 'C_W050', 'D_W100']}
    for r in all_results:
        n_any = sum(e['n_test'] for e in r['wave2'].values())
        if n_any == 0:
            continue
        n_seeds_wave2 += 1
        a_tot = sum(e['A_total_abs_err'] for e in r['wave2'].values() if e['A_total_abs_err'] is not None)
        for l in ['B_W025', 'C_W050', 'D_W100']:
            key = f'{l}_total_abs_err'
            tot = sum(e[key] for e in r['wave2'].values() if e[key] is not None)
            if tot < a_tot:
                wave2_better[l] += 1
    for l in ['B_W025', 'C_W050', 'D_W100']:
        print(f'second-wave toplam hata {l} iyi: {wave2_better[l]}/{n_seeds_wave2}')

    print('\n=== OZEL NESIL ALT-GRUP RAPORU (8 seed) ===')
    for s in all_special:
        print(f'seed {s["seed"]}: {s}')

    print('\n=== PROBE (sabit girdi, 4 model - son seed X_train kategorileri kullanildi) ===')
    # son seed'in egitilmis modellerini/lookup'unu tekrar kullan (ayni kod, tek sefer daha - probe icin)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEEDS[-1])
    pilot_X, pilot_y = prep_synth(pilot_raw, X_train.columns)
    wave2_X, wave2_y = prep_synth(wave2_raw, X_train.columns)
    X_train_c, pilot_X_c, wave2_X_c = to_cat(X_train, pilot_X, wave2_X)
    X_train_hp, _ = hp.attach_oof_feature(X_train_c, y_train)
    lookup_real = hp.build_price_lookup(X_train_c, y_train)
    pilot_hp = hp.attach_lookup_feature(pilot_X_c, lookup_real)
    wave2_hp = hp.attach_lookup_feature(wave2_X_c, lookup_real)
    hp_support_real = build_support_summary(X_train_c)

    w_real = pd.Series(1.0, index=X_train_hp.index)
    w_pilot = pd.Series(PILOT_WEIGHT, index=range(len(pilot_hp)))
    X_a = pd.concat([X_train_hp, pilot_hp], ignore_index=True)
    y_a = pd.concat([y_train.reset_index(drop=True), pilot_y], ignore_index=True)
    w_a = pd.concat([w_real.reset_index(drop=True), w_pilot], ignore_index=True)
    probe_models = {}
    m_a = LGBMRegressor(**BASELINE_PARAMS); m_a.fit(X_a, y_a, sample_weight=w_a); probe_models['A'] = m_a
    for label, w in WEIGHT_MODELS:
        X_b = pd.concat([X_a, wave2_hp], ignore_index=True)
        y_b = pd.concat([y_a, wave2_y], ignore_index=True)
        w_b = pd.concat([w_a, pd.Series([w] * len(wave2_hp))], ignore_index=True)
        m = LGBMRegressor(**BASELINE_PARAMS); m.fit(X_b, y_b, sample_weight=w_b)
        probe_models[label] = m

    reference_year = PRICE_REFERENCE_DATE.year
    for marka, model, yil, km, kasa, renk, mh, mg in PROBE_CASES:
        yas = max(reference_year - yil, 0)
        km_yil = km / (yas if yas > 0 else 1)
        hp_val, hp_src, hp_n = hp.lookup_price(marka, model, yas, lookup_real)
        row = pd.DataFrame([{
            'marka': marka, 'model': model, 'paket': 'Belirtilmemiş', 'kasa_turu': kasa, 'renk': renk,
            'motor_hacmi': mh, 'motor_gucu': mg, 'yil': yil, 'kilometre': km, 'yakit_turu': 'Benzin',
            'vites': 'Otomatik', 'degisen_sayisi': 0, 'boyali_sayisi': 0, 'agir_hasarli': 0,
            'degisen_sayisi_bilinmiyor': 0, 'boyali_sayisi_bilinmiyor': 0, 'yas': yas, 'km_yil': km_yil,
        }])
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype('category').cat.set_categories(X_train_hp[c].cat.categories)
        row = row.reindex(columns=X_train_hp.columns)
        row[hp.FEATURE_COLUMN] = hp_val
        peer_count, model_count, pct, peer_group = lookup_support(marka, model, mg, hp_support_real)
        conf = compute_confidence(peer_count, model_count, peer_group)
        print(f'\n{marka} {model} ({yil}, {km:,}km): hp_reference={hp_val:,.0f} hp_source={hp_src} '
              f'real_support={model_count} confidence={conf}')
        for label, m in probe_models.items():
            pred = float(m.predict(row)[0])
            pct_err = 100 * abs(pred - hp_val) / hp_val
            print(f'  {label:<10} pred={pred:,.0f}  pct_err_vs_hp_ref={pct_err:.1f}%')


if __name__ == '__main__':
    main()
