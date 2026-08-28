"""Faz 32 - production probe/regression suite. Mevcut (kabul edilmis) production
artifact davranisini tests/fixtures/production_probe_baseline.json fixture'ina
gore SABITLER - exact TL degeri hardcode ETMEZ, kabul edilmis tahminin etrafinda
bir TOLERANS bandi kullanir (bkz. generate_probe_baseline.py).

Bu dosya SADECE OKUR - hicbir retrain/sentetik uretim/artifact degisikligi
YAPMAZ. Fixture'i yeniden uretmek icin: python generate_probe_baseline.py
(bilinçli bir "yeni davranisi kabul ediyorum" karari GEREKTIRIR - CI'da
OTOMATIK calistirilmamali).

Calistirma (ai-model/ calisma dizini olarak):
    python -m unittest test_production_regression.py
"""
import json
import math
import os
import unittest

import joblib
import pandas as pd

import serve
from serve import PredictRequest, collect_category_errors, predict
from hierarchical_price import lookup_price
from hp_support import lookup_support, compute_confidence
from preprocess import TRAIN_PATH

BASE_DIR = os.path.dirname(__file__)
FIXTURE_PATH = os.path.join(BASE_DIR, 'tests', 'fixtures', 'production_probe_baseline.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')
WAVE30_PATH = os.path.join(BASE_DIR, '..', 'data', 'output', 'synthetic_pilot.csv')
WAVE31_PATH = os.path.join(BASE_DIR, '..', 'data', 'output', 'synthetic_second_wave_preview.csv')

EXPECTED_WAVE30_ROWS = 18
EXPECTED_WAVE31_ROWS = 38
EXPECTED_TOTAL_SYNTHETIC = 56
EXPECTED_WEIGHT = 0.50
EXPECTED_WAVE30_GROUPS = {'Ferrari|458', 'Lamborghini|Huracan', 'Rolls-Royce|Ghost'}
EXPECTED_WAVE31_GROUPS = {'Dodge|Ram', 'Rolls-Royce|Wraith', 'Lexus|LS', 'Aston Martin|Vantage',
                           'Bentley|Flying Spur', 'Mercedes - Benz|Maybach S', 'Mercedes - Benz|V Serisi'}

DODGE_RAM_LEGACY_ID = 'arabam-40745482'
LEXUS_LS_LEGACY_ID = 'arabam-41392003'
FLYING_SPUR_GEN2 = {'kaggle-ab-272', 'kaggle-ab-271', 'arabam-42433249', 'arabam-39563948'}
FLYING_SPUR_GEN3 = {'arabam-39320430', 'kaggle-ar-30608246', 'arabam-40722555'}


def load_fixture():
    with open(FIXTURE_PATH, encoding='utf-8') as f:
        return json.load(f)


def run_probe(payload):
    """Tahmin GERCEK /predict() akisindan (tam dogrulama/pipeline sadakati);
    hp_reference/hp_source/hp_support/confidence AYRI ve DOGRUDAN hesaplanir
    (hierarchical_price.lookup_price + hp_support.lookup_support) - boylece
    OTOMETRIK_DEBUG env degiskenine veya modul import SIRASINA (baska bir test
    dosyasi serve'i debug KAPALIYKEN ONCE import etmis olabilir - modul-seviyesi
    DEBUG_MODE sabiti bir kere okunur, sonradan degismez) hic BAGIMLI DEGILDIR."""
    req = PredictRequest(**payload)
    _, resolved = collect_category_errors(req)
    marka = resolved.get('marka') or req.brand
    resp = predict(req)
    yas = max(serve.MODEL_REFERENCE_YEAR - req.year, 0)
    hp_val, hp_src, _ = lookup_price(marka, req.model, yas, serve.HIERARCHICAL_PRICE_LOOKUP)
    peer_count, model_count, _, peer_group = lookup_support(marka, req.model, req.enginePower, serve.HP_SUPPORT)
    conf = compute_confidence(peer_count, model_count, peer_group)
    return {
        'prediction': resp['price'],
        'hp_reference': hp_val,
        'hp_source': hp_src,
        'hp_support': model_count,
        'confidence': conf,
    }


class ProductionRegressionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(FIXTURE_PATH):
            raise unittest.SkipTest(
                f'{FIXTURE_PATH} yok - once python generate_probe_baseline.py calistirilmali')
        cls.fixture = load_fixture()
        cls.artifact = joblib.load(MODEL_PATH)


class TestMainProbeRegression(ProductionRegressionTestBase):
    """10 ana grup - A) sanity, B) hierarchical price, C) confidence, D/E) kabul
    edilmis bant regresyonu."""

    def test_probe_prediction_is_finite_positive(self):
        for entry in self.fixture['main_probes']:
            with self.subTest(group=f"{entry['marka']}|{entry['model']}"):
                result = run_probe(entry['input'])
                pred = result['prediction']
                self.assertTrue(math.isfinite(pred), f'prediction finite degil: {pred}')
                self.assertGreater(pred, 0, f'prediction > 0 degil: {pred}')

    def test_probe_hp_source(self):
        for entry in self.fixture['main_probes']:
            with self.subTest(group=f"{entry['marka']}|{entry['model']}"):
                result = run_probe(entry['input'])
                self.assertEqual(result['hp_source'], 'brand_model',
                                  f"hp_source degisti: beklenen brand_model, gelen {result['hp_source']}")
                self.assertTrue(math.isfinite(result['hp_reference']))
                self.assertGreater(result['hp_reference'], 0)

    def test_probe_real_support_not_inflated(self):
        """Sentetik satirlarin support sayisini SISIRMEDIGINI dogrular - fixture'daki
        accepted_hp_support, generate_probe_baseline.py'de KENDI 'expected_real_support'una
        karsi ZATEN assert edilmisti; burada canli /predict akisinin AYNI degeri hala
        urettigi (regresyon YOK) dogrulanir."""
        for entry in self.fixture['main_probes']:
            with self.subTest(group=f"{entry['marka']}|{entry['model']}"):
                result = run_probe(entry['input'])
                self.assertEqual(result['hp_support'], entry['accepted_hp_support'],
                                  f"support degisti: kabul edilmis {entry['accepted_hp_support']}, "
                                  f"simdi {result['hp_support']} - sentetik satirlar support'u sisirmis olabilir")

    def test_probe_confidence_remains_low(self):
        for entry in self.fixture['main_probes']:
            with self.subTest(group=f"{entry['marka']}|{entry['model']}"):
                result = run_probe(entry['input'])
                self.assertEqual(result['confidence'], 'low',
                                  f"confidence degisti: kabul edilmis 'low', simdi '{result['confidence']}'")

    def test_probe_prediction_within_accepted_band(self):
        for entry in self.fixture['main_probes']:
            with self.subTest(group=f"{entry['marka']}|{entry['model']}"):
                result = run_probe(entry['input'])
                pred = result['prediction']
                accepted = entry['accepted_prediction']
                tol = entry['allowed_prediction_delta_pct'] / 100.0
                lo, hi = accepted * (1 - tol), accepted * (1 + tol)
                self.assertTrue(
                    lo <= pred <= hi,
                    f"prediction kabul bandi disinda: {pred:,.0f} not in [{lo:,.0f}, {hi:,.0f}] "
                    f"(accepted={accepted:,.0f}, tol=±{entry['allowed_prediction_delta_pct']}%, "
                    f"neden: {entry['tolerance_reason']})",
                )


class TestSpecialGenerationProbes(ProductionRegressionTestBase):
    """Nesil-izolasyon sanity probe'lari - ratio/band regresyonuna TABI DEGIL,
    sadece finite/>0 kontrolu (bkz. generate_probe_baseline.py SPECIAL_PROBES)."""

    def test_special_probe_sanity(self):
        for entry in self.fixture['special_probes']:
            with self.subTest(label=entry['label']):
                result = run_probe(entry['input'])
                self.assertTrue(math.isfinite(result['prediction']))
                self.assertGreater(result['prediction'], 0)


class TestDodgeRamGenerationIsolation(ProductionRegressionTestBase):
    def test_2004_legacy_not_used_as_synthetic_parent(self):
        wave31 = pd.read_csv(WAVE31_PATH)
        hits = wave31[wave31['source_parent_ids'].str.contains(DODGE_RAM_LEGACY_ID, na=False)]
        self.assertEqual(len(hits), 0, f'{DODGE_RAM_LEGACY_ID} (2004 legacy Ram) sentetik parent olarak kullanilmis!')

    def test_dodge_ram_synthetic_row_count(self):
        wave31 = pd.read_csv(WAVE31_PATH)
        n = len(wave31[(wave31['marka'] == 'Dodge') & (wave31['model'] == 'Ram')])
        self.assertEqual(n, 3, f'Dodge Ram sentetik satir sayisi degisti: beklenen 3, bulunan {n}')


class TestLexusLSGenerationIsolation(ProductionRegressionTestBase):
    def test_2015_legacy_not_used_as_synthetic_parent(self):
        wave31 = pd.read_csv(WAVE31_PATH)
        hits = wave31[wave31['source_parent_ids'].str.contains(LEXUS_LS_LEGACY_ID, na=False)]
        self.assertEqual(len(hits), 0, f'{LEXUS_LS_LEGACY_ID} (2015 eski nesil LS) sentetik parent olarak kullanilmis!')


class TestFlyingSpurGenerationIsolation(ProductionRegressionTestBase):
    def test_no_cross_generation_synthetic_pairs(self):
        wave31 = pd.read_csv(WAVE31_PATH)
        fs = wave31[(wave31['marka'] == 'Bentley') & (wave31['model'] == 'Flying Spur')]
        for _, row in fs.iterrows():
            ids = row['source_parent_ids'].split('+')
            in_gen2 = any(i in FLYING_SPUR_GEN2 for i in ids)
            in_gen3 = any(i in FLYING_SPUR_GEN3 for i in ids)
            with self.subTest(ilan_id=row['ilan_id']):
                self.assertFalse(in_gen2 and in_gen3, f"{row['ilan_id']} gen2 VE gen3 parent'lari CAPRAZLIYOR: {ids}")


class TestAudiRSNegativeControl(ProductionRegressionTestBase):
    def test_audi_rs_has_no_synthetic_rows_in_csv(self):
        wave31 = pd.read_csv(WAVE31_PATH)
        n = len(wave31[(wave31['marka'] == 'Audi') & (wave31['model'] == 'RS')])
        self.assertEqual(n, 0, f'Audi RS icin {n} sentetik satir bulundu - bu grup HARIC tutulmus olmaliydi')

    def test_audi_rs_not_in_artifact_wave_groups(self):
        groups = set(self.artifact['synthetic_waves']['wave31']['groups']) | \
            set(self.artifact['synthetic_waves']['wave30']['groups'])
        self.assertNotIn('Audi|RS', groups)


class TestSyntheticMetadataIntegrity(ProductionRegressionTestBase):
    def test_synthetic_enabled_and_weight(self):
        self.assertTrue(self.artifact['synthetic_enabled'])
        self.assertEqual(self.artifact['synthetic_weight'], EXPECTED_WEIGHT)
        self.assertEqual(self.artifact['synthetic_total_rows'], EXPECTED_TOTAL_SYNTHETIC)

    def test_wave_counts_and_groups(self):
        w30 = self.artifact['synthetic_waves']['wave30']
        w31 = self.artifact['synthetic_waves']['wave31']
        self.assertEqual(w30['row_count'], EXPECTED_WAVE30_ROWS)
        self.assertEqual(set(w30['groups']), EXPECTED_WAVE30_GROUPS)
        self.assertEqual(w31['row_count'], EXPECTED_WAVE31_ROWS)
        self.assertEqual(set(w31['groups']), EXPECTED_WAVE31_GROUPS)

    def test_csv_row_counts_match_metadata(self):
        wave30 = pd.read_csv(WAVE30_PATH)
        wave31 = pd.read_csv(WAVE31_PATH)
        self.assertEqual(len(wave30), EXPECTED_WAVE30_ROWS)
        self.assertEqual(len(wave31), EXPECTED_WAVE31_ROWS)


class TestSupportInvariant(ProductionRegressionTestBase):
    """Faz32 SUPPORT INVARIANT: artifact hierarchical_price'daki n == hp_support
    model_stats count == canli /predict confidence.model_count == beklenen GERCEK
    support - UC KAYNAK da AYNI olmali, sentetik hicbir yerde support'u artirmamali."""

    EXPECTED_REAL_SUPPORT = {
        'Ferrari|458': 7, 'Lamborghini|Huracan': 2, 'Rolls-Royce|Ghost': 6,
        'Dodge|Ram': 4, 'Rolls-Royce|Wraith': 6, 'Lexus|LS': 5, 'Aston Martin|Vantage': 6,
        'Bentley|Flying Spur': 7, 'Mercedes - Benz|Maybach S': 6, 'Mercedes - Benz|V Serisi': 9,
    }

    def test_three_sources_agree(self):
        hp_lookup = self.artifact['hierarchical_price']
        hp_support = self.artifact['hp_support']
        for key, expected in self.EXPECTED_REAL_SUPPORT.items():
            marka, model = key.split('|')
            with self.subTest(group=key):
                curve = hp_lookup['brand_model_curve'].get(f'{marka}\x1f{model}')
                stats = hp_support['model_stats'].get(f'{marka}\x1f{model}')
                self.assertIsNotNone(curve, f'{key} hierarchical_price brand_model_curve icinde yok')
                self.assertIsNotNone(stats, f'{key} hp_support model_stats icinde yok')
                self.assertEqual(curve['n'], expected, f'{key}: hierarchical_price n={curve["n"]} != beklenen {expected}')
                self.assertEqual(stats['count'], expected, f'{key}: hp_support count={stats["count"]} != beklenen {expected}')
                self.assertEqual(curve['n'], stats['count'], f'{key}: iki kaynak birbirini tutmuyor')


class TestTrainDatasetIntegrity(ProductionRegressionTestBase):
    def test_train_dataset_contains_no_synthetic_rows(self):
        df = pd.read_csv(TRAIN_PATH, low_memory=False, usecols=['ilan_id'])
        synthetic_like = df[df['ilan_id'].astype(str).str.startswith('synthetic')]
        self.assertEqual(len(synthetic_like), 0,
                          f'train_dataset.csv icinde {len(synthetic_like)} sentetik-benzeri satir bulundu!')


if __name__ == '__main__':
    unittest.main()
