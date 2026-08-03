"""Faz 20: hierarchical_price.py (brand_model_median_price) icin birim
regresyon testleri. Kucuk sentetik veriyle calisir - egitilmis production
artefaktina bagimli DEGILDIR (bkz. test_serve.py'deki entegrasyon testleri
icin gercek artefakt kullanan testler).

Calistirma (ai-model/ calisma dizini olarak):
    python -m unittest test_hierarchical_price.py
"""
import unittest

import numpy as np
import pandas as pd

from hierarchical_price import (
    FEATURE_COLUMN, OOF_SEED, attach_lookup_feature, attach_oof_feature,
    build_price_lookup, compute_oof_feature, lookup_price,
)


def toy_X():
    return pd.DataFrame({
        'marka': ['Ford', 'Ford', 'Ford', 'Ford', 'BMW', 'BMW', 'Renault'],
        'model': ['Focus', 'Focus', 'Focus', 'Fiesta', '3 Serisi', '3 Serisi', 'Clio'],
    })


def toy_y():
    # Focus icin asimetrik degerler bilerek secildi (400k/900k/950k) - 2'li alt
    # kumelerin ortalamasi (fold-train medyanlari) TAM (fold'suz) medyanla
    # (900k) COT eslesmesin diye (bkz. test_oof_value_never_uses_own_row_full_group_median).
    return pd.Series([400_000, 900_000, 950_000, 300_000, 900_000, 950_000, 250_000])


class TestFallbackChain(unittest.TestCase):
    """Faz 23: 4 katmanli fallback senaryolari: bilinen marka-model ->
    dogrudan brand_model medyani; bilinmeyen marka + bilinen model -> model
    (markadan bagimsiz) fallback; bilinmeyen model + bilinen marka -> marka
    fallback; ikisi de bilinmeyen -> global fallback."""

    def setUp(self):
        self.lookup = build_price_lookup(toy_X(), toy_y())

    def test_known_brand_model_uses_direct_median(self):
        value, source = lookup_price('Ford', 'Focus', self.lookup)
        self.assertEqual(source, 'brand_model')
        self.assertEqual(value, np.median([400_000, 900_000, 950_000]))

    def test_unseen_brand_known_model_falls_back_to_model_tier_combined_across_brands(self):
        """Faz 23'un yeni katmani: marka hic gorulmemis ama MODEL adi baska
        marka(lar) altinda gorulmusse, o modelin TUM markalar birlestirilerek
        hesaplanan medyanina duser - global'e degil."""
        X = pd.DataFrame({'marka': ['Ford', 'Toyota'], 'model': ['Corolla', 'Corolla']})
        y = pd.Series([600_000, 650_000])
        lookup = build_price_lookup(X, y)
        value, source = lookup_price('Honda', 'Corolla', lookup)
        self.assertEqual(source, 'model')
        self.assertEqual(value, np.median([600_000, 650_000]))

    def test_unknown_model_known_brand_falls_back_to_brand(self):
        value, source = lookup_price('Ford', 'Puma (Ford de gormedi)', self.lookup)
        self.assertEqual(source, 'brand')
        self.assertEqual(value, np.median([400_000, 900_000, 950_000, 300_000]))

    def test_unknown_brand_falls_back_to_global(self):
        value, source = lookup_price('Toyota', 'Corolla', self.lookup)
        self.assertEqual(source, 'global')
        self.assertEqual(value, float(toy_y().median()))

    def test_single_sample_group_still_gets_own_median_no_count_threshold(self):
        """Renault Clio tek ornekli bir grup - yine de kendi (tek satirlik)
        medyanini alir, marka/global'e degil (bkz. modul docstring'i - count
        esigi YOK, ablation'la BIREBIR AYNI davranis)."""
        value, source = lookup_price('Renault', 'Clio', self.lookup)
        self.assertEqual(source, 'brand_model')
        self.assertEqual(value, 250_000)

    def test_attach_lookup_feature_matches_row_by_row_lookup(self):
        X_new = pd.DataFrame({
            'marka': ['Ford', 'Ford', 'Toyota'],
            'model': ['Focus', 'Bilinmeyen', 'Corolla'],
        })
        attached = attach_lookup_feature(X_new, self.lookup)
        self.assertEqual(attached[FEATURE_COLUMN].iloc[0], np.median([400_000, 900_000, 950_000]))
        self.assertEqual(attached[FEATURE_COLUMN].iloc[1], np.median([400_000, 900_000, 950_000, 300_000]))
        self.assertEqual(attached[FEATURE_COLUMN].iloc[2], float(toy_y().median()))


class TestLeakagePrevention(unittest.TestCase):
    """5-fold OOF: bir egitim satirinin feature degeri kendi fiyatini
    HESABA KATAN bir gruptan gelmemeli."""

    def test_oof_value_never_uses_own_row_full_group_median(self):
        X, y = toy_X(), toy_y()
        oof_values, oof_sources = compute_oof_feature(X, y)
        full_bmm = X.assign(fiyat=y.values).groupby(['marka', 'model'], observed=True)['fiyat'].median()

        # Ford Focus grubunda (3 satir) TAM (fold'suz) medyan 900.000 - eger
        # OOF dogru calismiyorsa (sizinti varsa), her satirin OOF degeri bu
        # TAM medyanla birebir ayni olurdu. Kucuk (n=7, 5-fold) ornek boyutu
        # nedeniyle fold'lar arasinda bazen ayni deger cikabilir (2 satirlik
        # bir fold_train'de Focus grubu tek satira duserse) - bu yuzden
        # KESIN esitsizlik yerine, en az bir satirin farkli oldugu (sizinti
        # OLMADIGININ somut kaniti) dogrulanir.
        focus_mask = (X['marka'] == 'Ford') & (X['model'] == 'Focus')
        focus_oof = oof_values[focus_mask.values]
        full_focus_median = full_bmm[('Ford', 'Focus')]
        self.assertTrue(
            np.any(focus_oof != full_focus_median),
            'OOF degerleri TAM (sizintili) medyanla birebir ayni - out-of-fold hesaplama calismiyor olabilir',
        )

    def test_oof_source_is_diagnostic_and_covers_all_rows(self):
        X, y = toy_X(), toy_y()
        _, sources = compute_oof_feature(X, y)
        self.assertEqual(len(sources), len(X))
        self.assertTrue(set(sources).issubset({'brand_model', 'brand', 'global'}))


class TestDeterminism(unittest.TestCase):
    """Ayni girdi icin yeniden baslatmalar arasinda BIREBIR AYNI tahmin -
    build_price_lookup/compute_oof_feature'da rastgelelik SADECE sabit
    OOF_SEED ile KFold'da var, ikisi de deterministik olmali."""

    def test_build_price_lookup_is_deterministic(self):
        X, y = toy_X(), toy_y()
        lookup1 = build_price_lookup(X, y)
        lookup2 = build_price_lookup(X, y)
        self.assertEqual(lookup1['brand_model_median'], lookup2['brand_model_median'])
        self.assertEqual(lookup1['brand_median'], lookup2['brand_median'])
        self.assertEqual(lookup1['global_median'], lookup2['global_median'])

    def test_compute_oof_feature_is_deterministic_across_calls(self):
        X, y = toy_X(), toy_y()
        values1, _ = compute_oof_feature(X, y, seed=OOF_SEED)
        values2, _ = compute_oof_feature(X, y, seed=OOF_SEED)
        np.testing.assert_array_equal(values1, values2)

    def test_attach_oof_feature_reproducible_end_to_end(self):
        X, y = toy_X(), toy_y()
        X1, _ = attach_oof_feature(X, y)
        X2, _ = attach_oof_feature(X, y)
        pd.testing.assert_series_equal(X1[FEATURE_COLUMN], X2[FEATURE_COLUMN])


class TestLookupArtifactShape(unittest.TestCase):
    """Inference artefaktinin (build_price_lookup ciktisi) gorev talebindeki
    tum alanlari icermesi gerekir - versiyon, fallback zinciri, fold/seed,
    referans tarihi, normalizasyon notu."""

    def test_lookup_contains_required_metadata_fields(self):
        lookup = build_price_lookup(toy_X(), toy_y())
        for key in ('lookup_version', 'feature_column', 'fallback_chain', 'oof_n_splits',
                    'oof_seed', 'brand_model_median', 'model_median', 'brand_median', 'global_median',
                    'price_reference_date', 'training_data_hash', 'normalization_notes'):
            self.assertIn(key, lookup)
        self.assertEqual(lookup['fallback_chain'], ['brand_model', 'model', 'brand', 'global'])
        self.assertEqual(lookup['feature_column'], FEATURE_COLUMN)

    def test_training_data_hash_is_deterministic_and_content_sensitive(self):
        lookup1 = build_price_lookup(toy_X(), toy_y())
        lookup2 = build_price_lookup(toy_X(), toy_y())
        self.assertEqual(lookup1['training_data_hash'], lookup2['training_data_hash'])
        other_y = toy_y() + 1
        lookup3 = build_price_lookup(toy_X(), other_y)
        self.assertNotEqual(lookup1['training_data_hash'], lookup3['training_data_hash'])


if __name__ == '__main__':
    unittest.main()
