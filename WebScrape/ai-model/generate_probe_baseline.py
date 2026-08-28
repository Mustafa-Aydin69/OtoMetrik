"""Faz 32 - production_probe_baseline.json fixture'ini MEVCUT (kabul edilmis)
production artifact'tan uretir. Bu script SADECE SNAPSHOT alir - hicbir
retrain/sentetik uretim/artifact degisikligi YAPMAZ. Ciktisi bir kere
uretilip test fixture'i olarak SABIT kalmasi beklenir (bkz. tests/fixtures/).

10 ana probe (kabul edilmis production davranisini regression-test icin
sabitler) + 3 ozel nesil-izolasyon probe'u (Dodge Ram 2004 legacy, Lexus LS
2015 legacy, Bentley Flying Spur gen2) - bunlar SADECE sanity icin, ratio/
band regression kontrolune TABI DEGIL (bkz. test_production_regression.py).

Tahmin gercek serve.predict() akisindan (tam pipeline sadakati) alinir;
hp_reference/hp_source/hp_support/confidence ise DOGRUDAN hierarchical_price.
lookup_price() + hp_support.lookup_support() ile hesaplanir - boylece
OTOMETRIK_DEBUG env degiskenine veya modul import sirasina (bkz.
test_production_regression.py'nin AYNI run_probe() deseni) BAGIMLI DEGILDIR.

Calistirma (ai-model/ calisma dizini olarak): python generate_probe_baseline.py
"""
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import serve
from serve import PredictRequest, collect_category_errors, predict
from hierarchical_price import lookup_price
from hp_support import lookup_support, compute_confidence

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'production_probe_baseline.json')

DEFAULT_TOLERANCE_PCT = 25.0
WIDE_TOLERANCE_GROUPS = {
    'Dodge|Ram': (35.0, 'real support=4, en ince/en belirsiz grup - Faz31 pilot analizinde hp probe sapmasi '
                         '%38-64 araliginda olculdu, dar bant kucuk retrain gurultusunde bile patlardi'),
}

# 10 ana probe (bkz. Faz31 retrain dogrulama turundaki AYNI girdiler - tutarlilik icin degistirilmedi)
MAIN_PROBES = [
    dict(marka='Ferrari', model='458', expected_support=7,
         payload=dict(brand='Ferrari', model='458', year=2013, mileage=25000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Coupe', color='Kırmızı', engineDisplacement=4250, enginePower=563, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Lamborghini', model='Huracan', expected_support=2,
         payload=dict(brand='Lamborghini', model='Huracan', year=2016, mileage=30000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Coupe', color='Gri', engineDisplacement=5250, enginePower=610, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Rolls-Royce', model='Ghost', expected_support=6,
         payload=dict(brand='Rolls-Royce', model='Ghost', year=2014, mileage=50000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Siyah', engineDisplacement=6592, enginePower=563, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Dodge', model='Ram', expected_support=4,
         payload=dict(brand='Dodge', model='Ram', year=2023, mileage=15000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='SUV', color='Siyah', engineDisplacement=2750, enginePower=413, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Rolls-Royce', model='Wraith', expected_support=6,
         payload=dict(brand='Rolls-Royce', model='Wraith', year=2015, mileage=50000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Coupe', color='Siyah', engineDisplacement=6592, enginePower=632, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Lexus', model='LS', expected_support=5,
         payload=dict(brand='Lexus', model='LS', year=2021, mileage=50000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Siyah', engineDisplacement=3250, enginePower=363, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Aston Martin', model='Vantage', expected_support=6,
         payload=dict(brand='Aston Martin', model='Vantage', year=2012, mileage=50000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Coupe', color='Gri', engineDisplacement=4750, enginePower=438, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Bentley', model='Flying Spur', expected_support=7,
         payload=dict(brand='Bentley', model='Flying Spur', year=2021, mileage=42000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Siyah', engineDisplacement=3750, enginePower=538, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Mercedes - Benz', model='Maybach S', expected_support=6,
         payload=dict(brand='Mercedes - Benz', model='Maybach S', year=2018, mileage=120000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Siyah', engineDisplacement=3750, enginePower=463, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Mercedes - Benz', model='V Serisi', expected_support=9,
         payload=dict(brand='Mercedes - Benz', model='V Serisi', year=2021, mileage=80000, fuelType='Dizel', transmission='Otomatik',
                       bodyType='Camlı Van', color='Füme', engineDisplacement=1950, enginePower=237, trim='',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
]

# ozel nesil-izolasyon probe'lari - SANITY-ONLY (band/oran regression testine TABI DEGIL,
# bkz. modul docstring'i / kullanicinin gorev talebi Madde "2004 probe icin ... modern bandi kullanma")
SPECIAL_PROBES = [
    dict(marka='Dodge', model='Ram', label='dodge_ram_2004_legacy', sanity_only=True,
         note='2004 legacy Ram (kasa_turu=Hard top, dizel) - Faz31 sentetiginde HIC parent olarak kullanilmadi. '
              'Modern SUV kumesi bandiyla KARSILASTIRILMAZ, sadece prediction sanity (finite, >0) kontrol edilir.',
         payload=dict(brand='Dodge', model='Ram', year=2004, mileage=225000, fuelType='Dizel', transmission='Düz',
                       bodyType='Hard top', color='Gri (Gümüş)', engineDisplacement=5883, enginePower=230, trim='TD',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
    dict(marka='Lexus', model='LS', label='lexus_ls_2015_legacy', sanity_only=True,
         note='2015 eski nesil LS (4750.5cc, buyuk V8 hibrit) - Faz31 sentetiginde HIC parent olarak kullanilmadi '
              '(sadece 2018-2022 LS500h kumesi kullanildi). Modern bandiyla KARSILASTIRILMAZ, sadece sanity.',
         payload=dict(brand='Lexus', model='LS', year=2015, mileage=104177, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Siyah', engineDisplacement=4750.5, enginePower=438, trim='h',
                       replacedPartsCount=0, paintedPartsCount=4, heavyDamage=False)),
    dict(marka='Bentley', model='Flying Spur', label='bentley_flying_spur_gen2', sanity_only=True,
         note='2. nesil Flying Spur (2013-2014, 10.5-14.5M TL fiyat bandi) - ana probe (Bentley|Flying Spur, '
              'MAIN_PROBES) 3. nesli (2020-2022) temsil eder. Faz31 sentetiginde gen2/gen3 CAPRAZ cift '
              'KURULMADI - bu iki probe birbirinden BAGIMSIZ degerlendirilir, ortak bant kullanilmaz.',
         payload=dict(brand='Bentley', model='Flying Spur', year=2013, mileage=119000, fuelType='Benzin', transmission='Otomatik',
                       bodyType='Sedan', color='Füme', engineDisplacement=5750, enginePower=563, trim='6.0',
                       replacedPartsCount=0, paintedPartsCount=0, heavyDamage=False)),
]


def run_probe(marka, model, payload):
    req = PredictRequest(**payload)
    _, resolved = collect_category_errors(req)
    resolved_marka = resolved.get('marka') or req.brand
    resp = predict(req)
    yas = max(serve.MODEL_REFERENCE_YEAR - req.year, 0)
    hp_val, hp_src, _ = lookup_price(resolved_marka, req.model, yas, serve.HIERARCHICAL_PRICE_LOOKUP)
    peer_count, model_count, _, peer_group = lookup_support(resolved_marka, req.model, req.enginePower, serve.HP_SUPPORT)
    conf = compute_confidence(peer_count, model_count, peer_group)
    return {
        'prediction': resp['price'],
        'hp_reference': hp_val,
        'hp_source': hp_src,
        'hp_support': model_count,
        'confidence': conf,
    }


def main():
    fixture = {'generated_at_artifact_note': 'Faz32 - Faz31 retrain sonrasi production artifact snapshot',
               'main_probes': [], 'special_probes': []}

    print('=== ANA PROBE\'LAR (10 grup) ===')
    for p in MAIN_PROBES:
        result = run_probe(p['marka'], p['model'], p['payload'])
        assert result['hp_support'] == p['expected_support'], (
            f"{p['marka']} {p['model']}: beklenen support {p['expected_support']}, artefaktta {result['hp_support']}")
        ratio = result['prediction'] / result['hp_reference'] if result['hp_reference'] else None
        key = f"{p['marka']}|{p['model']}"
        tol_pct, tol_reason = WIDE_TOLERANCE_GROUPS.get(key, (DEFAULT_TOLERANCE_PCT, 'standart tolerans - yeterli/orta seviye gercek destek'))

        entry = {
            'marka': p['marka'], 'model': p['model'], 'input': p['payload'],
            'accepted_prediction': result['prediction'],
            'accepted_hp_reference': result['hp_reference'],
            'accepted_hp_support': result['hp_support'],
            'accepted_hp_source': result['hp_source'],
            'accepted_confidence': result['confidence'],
            'accepted_ratio': round(ratio, 4) if ratio else None,
            'allowed_prediction_delta_pct': tol_pct,
            'tolerance_reason': tol_reason,
            'notes': f'expected_real_support={p["expected_support"]}',
        }
        fixture['main_probes'].append(entry)
        print(f"{p['marka']} {p['model']}: pred={result['prediction']:,} hp_ref={result['hp_reference']:,.0f} "
              f"ratio={ratio:.3f} support={result['hp_support']} conf={result['confidence']} tol=±{tol_pct}%")

    print('\n=== OZEL NESIL PROBE\'LAR (sanity-only) ===')
    for p in SPECIAL_PROBES:
        result = run_probe(p['marka'], p['model'], p['payload'])
        entry = {
            'marka': p['marka'], 'model': p['model'], 'label': p['label'], 'sanity_only': True,
            'input': p['payload'],
            'accepted_prediction': result['prediction'],
            'accepted_hp_reference': result['hp_reference'],
            'accepted_hp_support': result['hp_support'],
            'accepted_hp_source': result['hp_source'],
            'accepted_confidence': result['confidence'],
            'notes': p['note'],
        }
        fixture['special_probes'].append(entry)
        print(f"[{p['label']}]: pred={result['prediction']:,} hp_ref={result['hp_reference']:,.0f} "
              f"support={result['hp_support']} conf={result['confidence']}")

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, 'w', encoding='utf-8') as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)
    print(f'\nFixture yazildi: {FIXTURE_PATH}')


if __name__ == '__main__':
    main()
