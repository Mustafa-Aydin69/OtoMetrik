"""Faz 32 - tek komutla production probe raporu. tests/fixtures/
production_probe_baseline.json'daki KABUL EDILMIS degerlere karsi canli
artefakti calistirir, tablo basar. Hicbir retrain/sentetik uretim/artifact
degisikligi YAPMAZ - SADECE OKUR ve RAPORLAR.

Exit code: kritik regresyon varsa (support sismesi, confidence degisimi,
metadata bozulmasi, bant disi tahmin, Audi RS sentetige dahil olmasi,
generation-isolation ihlali, NaN/inf) != 0.

Calistirma (ai-model/ calisma dizini olarak): python production_probe_report.py
"""
import json
import math
import os
import sys

import joblib
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import serve
from serve import PredictRequest, collect_category_errors, predict
from hierarchical_price import lookup_price
from hp_support import lookup_support, compute_confidence

BASE_DIR = os.path.dirname(__file__)
FIXTURE_PATH = os.path.join(BASE_DIR, 'tests', 'fixtures', 'production_probe_baseline.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')
WAVE31_PATH = os.path.join(BASE_DIR, '..', 'data', 'output', 'synthetic_second_wave_preview.csv')
DODGE_RAM_LEGACY_ID = 'arabam-40745482'
LEXUS_LS_LEGACY_ID = 'arabam-41392003'
FLYING_SPUR_GEN2 = {'kaggle-ab-272', 'kaggle-ab-271', 'arabam-42433249', 'arabam-39563948'}
FLYING_SPUR_GEN3 = {'arabam-39320430', 'kaggle-ar-30608246', 'arabam-40722555'}


def run_probe(payload):
    req = PredictRequest(**payload)
    _, resolved = collect_category_errors(req)
    marka = resolved.get('marka') or req.brand
    resp = predict(req)
    yas = max(serve.MODEL_REFERENCE_YEAR - req.year, 0)
    hp_val, hp_src, _ = lookup_price(marka, req.model, yas, serve.HIERARCHICAL_PRICE_LOOKUP)
    peer_count, model_count, _, peer_group = lookup_support(marka, req.model, req.enginePower, serve.HP_SUPPORT)
    conf = compute_confidence(peer_count, model_count, peer_group)
    return {'prediction': resp['price'], 'hp_reference': hp_val, 'hp_source': hp_src,
            'hp_support': model_count, 'confidence': conf}


def main():
    critical_failures = []

    if not os.path.exists(FIXTURE_PATH):
        print(f'HATA: fixture yok ({FIXTURE_PATH}) - once python generate_probe_baseline.py calistirin')
        return 1

    with open(FIXTURE_PATH, encoding='utf-8') as f:
        fixture = json.load(f)
    artifact = joblib.load(MODEL_PATH)

    print('=== ANA PROBE RAPORU ===')
    print(f"{'Grup':<32}{'Prediction':>14}{'Accepted':>14}{'Delta%':>9}{'HP ref':>14}{'Support':>8}{'Conf':>7}  Sonuc")
    for entry in fixture['main_probes']:
        key = f"{entry['marka']}|{entry['model']}"
        result = run_probe(entry['input'])
        pred, accepted = result['prediction'], entry['accepted_prediction']
        delta_pct = 100 * (pred - accepted) / accepted
        tol = entry['allowed_prediction_delta_pct']

        row_fail = []
        if not (math.isfinite(pred) and pred > 0):
            row_fail.append('NaN/inf veya <=0 prediction'); critical_failures.append(f'{key}: NaN/inf prediction')
        if abs(delta_pct) > tol:
            row_fail.append(f'bant disi (±{tol}%)'); critical_failures.append(f'{key}: prediction bant disi ({delta_pct:+.1f}%, izin ±{tol}%)')
        if result['hp_support'] != entry['accepted_hp_support']:
            row_fail.append('support degisti'); critical_failures.append(f'{key}: support {entry["accepted_hp_support"]}->{result["hp_support"]}')
        if result['confidence'] != entry['accepted_confidence']:
            row_fail.append('confidence degisti'); critical_failures.append(f'{key}: confidence {entry["accepted_confidence"]}->{result["confidence"]}')
        if result['hp_source'] != 'brand_model':
            row_fail.append('hp_source != brand_model'); critical_failures.append(f'{key}: hp_source={result["hp_source"]}')

        status = 'PASS' if not row_fail else 'FAIL: ' + ', '.join(row_fail)
        print(f"{key:<32}{pred:>14,.0f}{accepted:>14,.0f}{delta_pct:>+8.1f}%{result['hp_reference']:>14,.0f}"
              f"{result['hp_support']:>8}{result['confidence']:>7}  {status}")

    print('\n=== OZEL NESIL PROBE\'LARI (sanity-only) ===')
    for entry in fixture['special_probes']:
        result = run_probe(entry['input'])
        ok = math.isfinite(result['prediction']) and result['prediction'] > 0
        if not ok:
            critical_failures.append(f"{entry['label']}: NaN/inf/<=0 prediction")
        print(f"{entry['label']:<32} pred={result['prediction']:>14,.0f}  support={result['hp_support']}  "
              f"conf={result['confidence']}  {'PASS' if ok else 'FAIL'}")

    print('\n=== GENERATION-ISOLATION KONTROLLERI ===')
    wave31 = pd.read_csv(WAVE31_PATH)
    ram_hits = wave31[wave31['source_parent_ids'].str.contains(DODGE_RAM_LEGACY_ID, na=False)]
    ls_hits = wave31[wave31['source_parent_ids'].str.contains(LEXUS_LS_LEGACY_ID, na=False)]
    fs = wave31[(wave31['marka'] == 'Bentley') & (wave31['model'] == 'Flying Spur')]
    fs_cross = 0
    for _, r in fs.iterrows():
        ids = r['source_parent_ids'].split('+')
        if any(i in FLYING_SPUR_GEN2 for i in ids) and any(i in FLYING_SPUR_GEN3 for i in ids):
            fs_cross += 1
    audi_rs_rows = len(wave31[(wave31['marka'] == 'Audi') & (wave31['model'] == 'RS')])

    for label, bad, msg in [
        ('Dodge Ram 2004 legacy parent kullanimi', len(ram_hits), 'legacy 2004 satiri sentetik parent olarak kullanilmis'),
        ('Lexus LS 2015 legacy parent kullanimi', len(ls_hits), 'legacy 2015 satiri sentetik parent olarak kullanilmis'),
        ('Flying Spur gen2/gen3 capraz cift', fs_cross, 'nesiller arasi capraz sentetik cift bulundu'),
        ('Audi RS sentetik satir', audi_rs_rows, 'Audi RS icin sentetik satir bulundu (HARIC tutulmus olmaliydi)'),
    ]:
        status = 'PASS' if bad == 0 else f'FAIL ({bad}): {msg}'
        if bad != 0:
            critical_failures.append(f'{label}: {msg} (n={bad})')
        print(f'{label:<45} {status}')

    print('\n=== METADATA INVARIANT ===')
    checks = [
        ('synthetic_enabled == True', artifact.get('synthetic_enabled') is True),
        ('synthetic_weight == 0.50', artifact.get('synthetic_weight') == 0.50),
        ('synthetic_total_rows == 56', artifact.get('synthetic_total_rows') == 56),
        ('wave30.row_count == 18', artifact.get('synthetic_waves', {}).get('wave30', {}).get('row_count') == 18),
        ('wave31.row_count == 38', artifact.get('synthetic_waves', {}).get('wave31', {}).get('row_count') == 38),
    ]
    for label, ok in checks:
        if not ok:
            critical_failures.append(f'metadata: {label} basarisiz')
        print(f'{label:<35} {"PASS" if ok else "FAIL"}')

    print(f'\n{"=" * 60}')
    if critical_failures:
        print(f'SONUC: {len(critical_failures)} KRITIK REGRESYON BULUNDU:')
        for f in critical_failures:
            print(f'  - {f}')
        return 1
    print('SONUC: kritik regresyon YOK, tum kontroller PASS.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
