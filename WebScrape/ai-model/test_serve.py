"""Faz 13: serve.py'nin kategori-eslestirme (category_mapping.py) davranisinin
regresyon testleri. Calistirma (ai-model/ calisma dizini olarak):
    python -m unittest test_serve.py

FastAPI TestClient kullanir - gercek bir uvicorn sureci baslatmadan, ayni
Python surecinde (in-process) /predict ve /categories'i cagirir.
"""
import unittest

from fastapi.testclient import TestClient

from domain_validation import FIELD_BOUNDS, MAX_TOTAL_DAMAGED_PARTS
from preprocess import CURRENT_YEAR
from serve import app

client = TestClient(app)

BASE_PAYLOAD = {
    "brand": "Ford",
    "model": "Focus",
    "year": 2018,
    "mileage": 85000,
    "fuelType": "Benzin",
    "transmission": "Otomatik",
    "bodyType": "Sedan",
    "color": "Beyaz",
    "engineDisplacement": 1600,
    "enginePower": 125,
    "trim": "Titanium",
    "replacedPartsCount": 1,
    "paintedPartsCount": 2,
    "heavyDamage": False,
}


def predict(payload):
    return client.post("/predict", json=payload)


class TestHealthAndCategories(unittest.TestCase):
    def test_health(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["model_loaded"])

    def test_categories_endpoint_lists_known_fields(self):
        resp = client.get("/categories")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for field in ("marka", "vites", "yakit_turu", "kasa_turu", "renk"):
            self.assertIn(field, body)
            self.assertGreater(len(body[field]["options"]), 0)


class TestLabelToCanonicalTranslation(unittest.TestCase):
    """Website etiketiyle gonderilen istek, canonical degerle dogrudan
    gonderilen istekle AYNI tahmini uretmeli (bkz. gorev tanimindaki test 5/6)."""

    def test_vites_manuel_equals_duz(self):
        label_resp = predict({**BASE_PAYLOAD, "transmission": "Manuel"})
        canonical_resp = predict({**BASE_PAYLOAD, "transmission": "Düz"})
        self.assertEqual(label_resp.status_code, 200)
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(label_resp.json()["price"], canonical_resp.json()["price"])

    def test_marka_mercedes_benz_equals_mercedes_dash_benz(self):
        payload = {**BASE_PAYLOAD, "model": "C Serisi", "engineDisplacement": 2000, "enginePower": 190}
        label_resp = predict({**payload, "brand": "Mercedes-Benz"})
        canonical_resp = predict({**payload, "brand": "Mercedes - Benz"})
        self.assertEqual(label_resp.status_code, 200)
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(label_resp.json()["price"], canonical_resp.json()["price"])

    def test_yakit_turu_lpg_equals_lpg_and_benzin(self):
        label_resp = predict({**BASE_PAYLOAD, "fuelType": "LPG"})
        canonical_resp = predict({**BASE_PAYLOAD, "fuelType": "LPG & Benzin"})
        self.assertEqual(label_resp.status_code, 200)
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(label_resp.json()["price"], canonical_resp.json()["price"])

    def test_renk_gumus_equals_gri_gumus(self):
        label_resp = predict({**BASE_PAYLOAD, "color": "Gümüş"})
        canonical_resp = predict({**BASE_PAYLOAD, "color": "Gri (Gümüş)"})
        self.assertEqual(label_resp.status_code, 200)
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(label_resp.json()["price"], canonical_resp.json()["price"])

    def test_kasa_turu_hatchback_3_kapi_equals_canonical(self):
        label_resp = predict({**BASE_PAYLOAD, "bodyType": "Hatchback (3 Kapı)"})
        canonical_resp = predict({**BASE_PAYLOAD, "bodyType": "Hatchback/3"})
        self.assertEqual(label_resp.status_code, 200)
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(label_resp.json()["price"], canonical_resp.json()["price"])

    def test_marka_diger_resolves_to_missing_not_rejected(self):
        resp = predict({**BASE_PAYLOAD, "brand": "Diğer"})
        self.assertEqual(resp.status_code, 200)


class TestUnknownCategoryRejected(unittest.TestCase):
    """Ne etiket ne kanonik olarak taninan bir deger sessizce NaN'a
    dusurulmemeli - 422 ile, alan/deger/allowed_examples icerecek bicimde
    reddedilmeli."""

    def test_unknown_marka_returns_422_with_detail(self):
        resp = predict({**BASE_PAYLOAD, "brand": "Wroom Motors"})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["field"], "marka")
        self.assertEqual(detail["value"], "Wroom Motors")
        self.assertIn("allowed_examples", detail)
        self.assertGreater(len(detail["allowed_examples"]), 0)

    def test_unknown_vites_returns_422_with_detail(self):
        resp = predict({**BASE_PAYLOAD, "transmission": "Yürüyerek"})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["field"], "vites")
        self.assertEqual(detail["value"], "Yürüyerek")

    def test_ambiguous_bare_hatchback_is_rejected_not_guessed(self):
        """'Hatchback' (kapi sayisi belirtilmemis) UNSUPPORTED_LABELS'ta -
        otomatik Hatchback/3 veya /5'e eslenmemeli, 422 donmeli."""
        resp = predict({**BASE_PAYLOAD, "bodyType": "Hatchback"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["field"], "kasa_turu")

    def test_unsupported_minivan_is_rejected(self):
        resp = predict({**BASE_PAYLOAD, "bodyType": "Minivan"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["field"], "kasa_turu")


class TestDomainValidation(unittest.TestCase):
    """Faz 14: egitim-verisi-turevli alan sinirlari ve alanlar-arasi
    tutarlilik kurallari. Tum hatalar {code, field, value, message,
    allowed_range} sekliyle 422 doner."""

    def test_negative_mileage_rejected(self):
        resp = predict({**BASE_PAYLOAD, "mileage": -50000})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["code"], "invalid_value")
        self.assertEqual(detail["field"], "kilometre")

    def test_year_1900_rejected(self):
        resp = predict({**BASE_PAYLOAD, "year": 1900})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["field"], "yil")
        self.assertEqual(detail["allowed_range"], FIELD_BOUNDS["yil"])

    def test_future_year_rejected(self):
        resp = predict({**BASE_PAYLOAD, "year": CURRENT_YEAR + 5})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["field"], "yil")

    def test_negative_engine_power_rejected(self):
        resp = predict({**BASE_PAYLOAD, "enginePower": -100})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["field"], "motor_gucu")

    def test_absurd_engine_power_rejected(self):
        resp = predict({**BASE_PAYLOAD, "enginePower": 99999})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["field"], "motor_gucu")
        self.assertEqual(detail["allowed_range"], FIELD_BOUNDS["motor_gucu"])

    def test_blank_brand_rejected(self):
        # trim="" ile izole edilir - aksi halde "paket marka/model bosken
        # kabul edilmez" kurali da tetiklenip birden fazla hata donerdi
        # (bkz. test_trim_without_brand_or_model_rejected, o senaryoyu
        # ayrica test eder).
        resp = predict({**BASE_PAYLOAD, "brand": "", "trim": ""})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["code"], "invalid_value")
        self.assertEqual(detail["field"], "marka")

    def test_whitespace_only_model_rejected(self):
        resp = predict({**BASE_PAYLOAD, "model": "   ", "trim": ""})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["code"], "invalid_value")
        self.assertEqual(detail["field"], "model")

    def test_damaged_parts_total_exceeding_13_rejected(self):
        resp = predict({**BASE_PAYLOAD, "replacedPartsCount": 7, "paintedPartsCount": 7})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["code"], "inconsistent_combination")
        self.assertEqual(detail["value"], 14)

    def test_trim_without_brand_or_model_rejected(self):
        """brand bos VE trim dolu -> iki gercek sorun birden: "marka bos
        olamaz" + "marka bosken paket kabul edilmez". Ikisi de gecerli,
        birlikte liste olarak donmeli."""
        resp = predict({**BASE_PAYLOAD, "brand": "", "trim": "Titanium"})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertIsInstance(detail, list)
        fields = [d["field"] for d in detail]
        self.assertIn("marka", fields)
        self.assertIn("paket", fields)

    def test_multiple_errors_returned_as_list(self):
        resp = predict({**BASE_PAYLOAD, "brand": "", "trim": "", "mileage": -1})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertIsInstance(detail, list)
        self.assertEqual(len(detail), 2)

    def test_normal_car_returns_200(self):
        resp = predict(BASE_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("price", body)
        self.assertEqual(body["warnings"], [])

    def test_boundary_but_valid_car_returns_200(self):
        """Sinirda ama gecerli: yil alt sinirda, motor_gucu ust sinirda,
        degisen+boyali toplaminin tam 13'unde (MAX_TOTAL_DAMAGED_PARTS)."""
        resp = predict({
            **BASE_PAYLOAD,
            "year": FIELD_BOUNDS["yil"]["min"],
            "mileage": FIELD_BOUNDS["kilometre"]["max"],
            "enginePower": FIELD_BOUNDS["motor_gucu"]["max"],
            "replacedPartsCount": 6,
            "paintedPartsCount": MAX_TOTAL_DAMAGED_PARTS - 6,
        })
        self.assertEqual(resp.status_code, 200)

    def test_suspicious_but_possible_combination_returns_warning(self):
        """Yeni model + cok yuksek km/yil - egitim verisinde nadir ama
        gozlemlenmis (P99.9=67.540 km/yil ustu). Tahmini ENGELLEMEMELI,
        sadece warning eklemeli."""
        resp = predict({**BASE_PAYLOAD, "year": CURRENT_YEAR - 1, "mileage": 200000})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["warnings"]), 1)
        self.assertEqual(body["warnings"][0]["code"], "unusual_mileage_for_year")


class TestHpConfidence(unittest.TestCase):
    """Faz 17: peer-support tabanli guven/warning mekanizmasi. BASE_PAYLOAD
    zaten Ford Focus 125 HP - yaygin model + tipik HP icin taban senaryodur."""

    def test_common_model_typical_hp_high_confidence_no_warning(self):
        resp = predict(BASE_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["confidence"]["level"], "high")
        self.assertEqual(body["warnings"], [])

    def test_common_model_extreme_hp_low_confidence_with_warning(self):
        resp = predict({**BASE_PAYLOAD, "enginePower": 500})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["confidence"]["level"], "low")
        codes = [w["code"] for w in body["warnings"]]
        self.assertIn("low_support_high_power_segment", codes)

    def test_rare_model_normal_hp_low_or_medium_confidence(self):
        resp = predict({**BASE_PAYLOAD, "brand": "Aston Martin", "model": "DB9",
                        "enginePower": 470, "engineDisplacement": 5900})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["confidence"]["level"], ("low", "medium"))

    def test_bmw_m_serisi_reduced_confidence_with_warning(self):
        resp = predict({**BASE_PAYLOAD, "brand": "BMW", "model": "M Serisi",
                        "enginePower": 276, "engineDisplacement": 3000, "bodyType": "Coupe"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # peer_count=22, model_count=58 -> HIGH esiginin (50) altinda, "medium"
        # (veri-turevli esiklerle "low" degil ama tam guven de degil) - her
        # ikisi de warning URETIR (yalnizca "high" warning'siz kalir).
        self.assertIn(body["confidence"]["level"], ("low", "medium"))
        self.assertGreater(len(body["warnings"]), 0)

    def test_hyundai_accent_601hp_not_422_but_low_confidence_strong_warning(self):
        """Kullanicinin acik talimati: 601 HP genel-gecer sekilde reddedilmemeli
        (bazi araclarda gecerli olabilir) - marka-model bagimli guvenilir bir
        fiziksel kural olmadigi icin daha guvenli davranis low confidence +
        warning'dir, 422 DEGIL (2000 fiziksel sinirinin altinda kaldigi icin
        zaten domain_validation da reddetmez)."""
        resp = predict({**BASE_PAYLOAD, "brand": "Hyundai", "model": "Accent", "enginePower": 601})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["confidence"]["level"], "low")
        self.assertEqual(body["confidence"]["peer_count"], 0)
        codes = [w["code"] for w in body["warnings"]]
        self.assertIn("low_support_high_power_segment", codes)

    def test_physically_impossible_hp_still_422(self):
        resp = predict({**BASE_PAYLOAD, "enginePower": 5000})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["field"], "motor_gucu")

    def test_smoke_test_price_unchanged_by_confidence_mechanism(self):
        """confidence/warning eklenmesi normal senaryolarin FIYATINI
        degistirmemeli - sadece response'a ek alan eklenir."""
        resp = predict(BASE_PAYLOAD)
        body = resp.json()
        self.assertEqual(body["price"], body["prediction"])
        self.assertGreater(body["price"], 0)

    def test_confidence_lookup_latency_is_negligible(self):
        """Madde 6 - request basina DataFrame taramasi YASAK. 50 ardisik
        /predict cagrisinin ortalama round-trip suresi FastAPI overhead'i
        dahil bile birkaç ms araliginda kalmali (DataFrame taramasi olsaydi
        onlarca/yuzlerce ms'ye cikardi)."""
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            predict(BASE_PAYLOAD)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / 50
        self.assertLess(avg_ms, 50, f"ortalama istek suresi {avg_ms:.2f}ms - beklenenden yuksek")


if __name__ == "__main__":
    unittest.main()
