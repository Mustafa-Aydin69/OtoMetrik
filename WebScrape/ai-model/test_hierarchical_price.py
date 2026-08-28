"""Faz 20/29: hierarchical_price.py (brand_model_median_price) icin birim
regresyon testleri. Kucuk sentetik veriyle calisir - egitilmis production
artefaktina bagimli DEGILDIR (bkz. test_serve.py'deki entegrasyon testleri
icin gercek artefakt kullanan testler).

Faz 29: deger hesaplama duz medyandan yas-farkindalikli Theil-Sen egriye
gecti (bkz. hierarchical_price.py modul docstring'i). Bu dosyadaki toy
verilerin cogu KASITLI olarak TEK bir 'yas' degeri kullanir (distinct_ages=1)
- bu, _fit_curve()'u her zaman DUZ (slope=0) fallback'a dusurur, yani eski
(Faz 20-28) medyan-tabanli testler ayni mantikla gecerli kalir (log/exp
round-trip nedeniyle assertEqual yerine assertAlmostEqual kullanilir).
TestAgeAwareCurve sinifi ise CESITLI yaslarla gercek Theil-Sen davranisini
(farkli yaslarda farkli tahmin) ayrica dogrular.

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

TOY_YAS = 5  # tum toy satirlar AYNI yasta - _fit_curve() daima duz (slope=0) medyana geriler


def toy_X():
    return pd.DataFrame({
        'marka': ['Ford', 'Ford', 'Ford', 'Ford', 'BMW', 'BMW', 'Renault'],
        'model': ['Focus', 'Focus', 'Focus', 'Fiesta', '3 Serisi', '3 Serisi', 'Clio'],
        'yas': [TOY_YAS] * 7,
    })


def toy_y():
    # Focus icin asimetrik degerler bilerek secildi (400k/900k/950k) - 2'li alt
    # kumelerin ortalamasi (fold-train medyanlari) TAM (fold'suz) medyanla
    # (900k) COT eslesmesin diye (bkz. test_oof_value_never_uses_own_row_full_group_median).
    return pd.Series([400_000, 900_000, 950_000, 300_000, 900_000, 950_000, 250_000])


class TestFallbackChain(unittest.TestCase):
    """Faz 23: 4 katmanli fallback senaryolari: bilinen marka-model ->
    dogrudan brand_model degeri; bilinmeyen marka + bilinen model -> model
    (markadan bagimsiz) fallback; bilinmeyen model + bilinen marka -> marka
    fallback; ikisi de bilinmeyen -> global fallback. TOY_YAS sabit oldugu
    icin deger her zaman duz medyana esittir (bkz. modul docstring'i)."""

    def setUp(self):
        self.lookup = build_price_lookup(toy_X(), toy_y())

    def test_known_brand_model_uses_direct_median(self):
        value, source, n = lookup_price('Ford', 'Focus', TOY_YAS, self.lookup)
        self.assertEqual(source, 'brand_model')
        self.assertEqual(n, 3)
        self.assertAlmostEqual(value, np.median([400_000, 900_000, 950_000]), places=4)

    def test_unseen_brand_known_model_falls_back_to_model_tier_combined_across_brands(self):
        """Faz 23'un yeni katmani: marka hic gorulmemis ama MODEL adi baska
        marka(lar) altinda gorulmusse, o modelin TUM markalar birlestirilerek
        hesaplanan medyanina duser - global'e degil."""
        X = pd.DataFrame({'marka': ['Ford', 'Toyota'], 'model': ['Corolla', 'Corolla'], 'yas': [TOY_YAS, TOY_YAS]})
        y = pd.Series([600_000, 650_000])
        lookup = build_price_lookup(X, y)
        value, source, n = lookup_price('Honda', 'Corolla', TOY_YAS, lookup)
        self.assertEqual(source, 'model')
        self.assertEqual(n, 2)
        self.assertAlmostEqual(value, np.median([600_000, 650_000]), places=4)

    def test_unknown_model_known_brand_falls_back_to_brand(self):
        value, source, n = lookup_price('Ford', 'Puma (Ford de gormedi)', TOY_YAS, self.lookup)
        self.assertEqual(source, 'brand')
        self.assertEqual(n, 4)
        self.assertAlmostEqual(value, np.median([400_000, 900_000, 950_000, 300_000]), places=4)

    def test_unknown_brand_falls_back_to_global(self):
        value, source, n = lookup_price('Toyota', 'Corolla', TOY_YAS, self.lookup)
        self.assertEqual(source, 'global')
        self.assertEqual(n, 7)
        self.assertAlmostEqual(value, float(toy_y().median()), places=4)

    def test_single_sample_group_still_gets_own_median_no_count_threshold(self):
        """Renault Clio tek ornekli bir grup - yine de kendi (tek satirlik)
        degerini alir, marka/global'e degil (bkz. modul docstring'i - count
        esigi YOK, ablation'la BIREBIR AYNI davranis; n=1 zaten MIN_CURVE_POINTS'in
        altinda oldugu icin daima duz deger doner)."""
        value, source, n = lookup_price('Renault', 'Clio', TOY_YAS, self.lookup)
        self.assertEqual(source, 'brand_model')
        self.assertEqual(n, 1)
        self.assertAlmostEqual(value, 250_000, places=4)

    def test_attach_lookup_feature_matches_row_by_row_lookup(self):
        X_new = pd.DataFrame({
            'marka': ['Ford', 'Ford', 'Toyota'],
            'model': ['Focus', 'Bilinmeyen', 'Corolla'],
            'yas': [TOY_YAS, TOY_YAS, TOY_YAS],
        })
        attached = attach_lookup_feature(X_new, self.lookup)
        self.assertAlmostEqual(attached[FEATURE_COLUMN].iloc[0], np.median([400_000, 900_000, 950_000]), places=4)
        self.assertAlmostEqual(attached[FEATURE_COLUMN].iloc[1], np.median([400_000, 900_000, 950_000, 300_000]), places=4)
        self.assertAlmostEqual(attached[FEATURE_COLUMN].iloc[2], float(toy_y().median()), places=4)


class TestAgeAwareCurve(unittest.TestCase):
    """Faz 29: yeterli veri/yas cesitliligi (n>=MIN_CURVE_POINTS, distinct_ages>=2)
    oldugunda Theil-Sen egrisi FARKLI yaslarda FARKLI (yasla azalan) deger uretmeli -
    duz medyandan farkli olarak."""

    def test_curve_predicts_lower_price_for_older_query_age(self):
        X = pd.DataFrame({
            'marka': ['Cadillac'] * 5,
            'model': ['Escalade'] * 5,
            'yas': [2, 5, 8, 11, 14],
        })
        # fiyat yasla azalan, gurultusuz bir egri - Theil-Sen'in yon/buyukluk
        # yakalayabildigini dogrulamak icin
        y = pd.Series([9_000_000, 6_500_000, 4_500_000, 3_200_000, 2_200_000])
        lookup = build_price_lookup(X, y)

        young, source_young, n_young = lookup_price('Cadillac', 'Escalade', 2, lookup)
        old, source_old, n_old = lookup_price('Cadillac', 'Escalade', 14, lookup)
        self.assertEqual(source_young, 'brand_model')
        self.assertEqual(source_old, 'brand_model')
        self.assertEqual(n_young, 5)
        self.assertEqual(n_old, 5)
        self.assertGreater(young, old, 'genc (dusuk yas) sorgu, yasli sorgudan daha yuksek fiyat almali')

    def test_insufficient_data_falls_back_to_flat_median(self):
        """n<MIN_CURVE_POINTS (=3) ise yas'tan BAGIMSIZ, sabit (medyan) deger doner."""
        X = pd.DataFrame({'marka': ['Niche'] * 2, 'model': ['Rare'] * 2, 'yas': [1, 20]})
        y = pd.Series([1_000_000, 3_000_000])
        lookup = build_price_lookup(X, y)
        v1, _, n1 = lookup_price('Niche', 'Rare', 1, lookup)
        v2, _, n2 = lookup_price('Niche', 'Rare', 20, lookup)
        self.assertEqual(n1, 2)
        self.assertEqual(n2, 2)
        self.assertAlmostEqual(v1, v2, places=4)
        self.assertAlmostEqual(v1, np.median([1_000_000, 3_000_000]), places=4)

    def test_same_age_all_rows_falls_back_to_flat_median_even_with_enough_points(self):
        """n>=MIN_CURVE_POINTS ama TUM yaslar ayniysa (distinct_ages=1) egim
        tanimsiz olur - duz medyana geriler."""
        X = pd.DataFrame({'marka': ['Flat'] * 4, 'model': ['Line'] * 4, 'yas': [7, 7, 7, 7]})
        y = pd.Series([500_000, 600_000, 700_000, 800_000])
        lookup = build_price_lookup(X, y)
        value, _, n = lookup_price('Flat', 'Line', 3, lookup)  # sorgu yasi FARKLI olsa bile
        self.assertEqual(n, 4)
        self.assertAlmostEqual(value, np.median([500_000, 600_000, 700_000, 800_000]), places=4)


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
            np.any(np.abs(focus_oof - full_focus_median) > 1e-6),
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
        self.assertEqual(lookup1['brand_model_curve'], lookup2['brand_model_curve'])
        self.assertEqual(lookup1['brand_curve'], lookup2['brand_curve'])
        self.assertEqual(lookup1['global_curve'], lookup2['global_curve'])

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
                    'oof_seed', 'min_curve_points', 'brand_model_curve', 'model_curve',
                    'brand_curve', 'global_curve', 'price_reference_date',
                    'training_data_hash', 'normalization_notes'):
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
