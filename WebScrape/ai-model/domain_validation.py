"""Faz 14: server-side domain validation - egitim verisinin gercek yuzdelik
dagilimlarindan turetilmis alan sinirlari + alanlar-arasi tutarlilik kurallari.
category_mapping.py'nin (Faz 13, "hangi deger dogru formatta") devami - burada
"hangi deger GERCEKCI" sorusu cevaplanir.

Sinir secimleri (287.595 satirlik temiz egitim verisi, preprocess.load_clean_train_dataset()
uzerinden hesaplandi - bkz. dagitima hazirlik/domain-validation analizindeki
yuzdelik tablosu):
- yil: [1950, bu yilin takvim yili + 1] - egitim verisi min=1951, P0.1=1985.
  UST SINIR preprocess.CURRENT_YEAR'dan (egitimde "yas" ozelligini hesaplamak
  icin kullanilan, MODELE gomulu sabit - degistirilmesi train/serve skew
  yaratir, retrain gerektirir) KASITLI OLARAK BAGIMSIZDIR; date.today().year
  ile dinamik hesaplanir, aksi halde sistem takvim CURRENT_YEAR'i astikca
  (orn. 2028'de) gecerli girdileri gereksiz yere reddetmeye baslardi. NOT:
  build_feature_row()'daki "yas = CURRENT_YEAR - yil" hesaplamasi ise HALA
  preprocess.CURRENT_YEAR kullaniyor (egitimle tutarli olmasi gerektigi icin
  kasitli) - bu, takvim yili CURRENT_YEAR'i astikca yeni araclarin "yas"inin
  serve-time'da yanlis (0'a sabitlenmis) hesaplanacagi ayri, daha derin bir
  sorun olusturur; duzeltmesi model yeniden egitimi gerektirir, bkz. hata
  taksonomisi raporundaki ilgili not.
- kilometre: [0, 1_000_000] - preprocess.py zaten bu ustu filtreliyor (Faz 8/9
  EDA karari, "veri kirliligi") - website'in eski 2.000.000 siniri gevsekti.
- motor_gucu: [1, 2000] - Faz 16 (hp_quality_analysis.py) + Faz 17 (hp_support.py)
  SONRASI YENIDEN TANIMLANDI: eskiden TEK bir sayi (800) hem "fiziksel olarak
  mumkun mu" hem "model buna guvenilir tahmin uretebilir mi" sorularini
  birlikte cevaplamaya calisiyordu - bunlar FARKLI sorular. 2000, SADECE
  fiziksel-imkansizlik sinirdir (gercek uretim hypercarlari ~1500-1600 HP
  araligini kapsayacak marj ile) - "99999", "130000" gibi acik veri girisi
  hatalarini reddeder. "Model bu HP'yi guvenilir tahmin eder mi" sorusu ARTIK
  bu sabit sayidan degil, hp_support.py'nin marka+model peer-destek
  yogunlugundan (serve.py /predict yanitindaki "confidence"/"warnings")
  cevaplanir - bkz. Faz 16 Madde 6 bulgusu: 400+ HP'de egitim R2=-3.18 (800
  siniri BUNU cozmuyordu, cunku sorun sinir degil ornek sayisidir).
- motor_hacmi: [0, 9000] (elektrikte 0 kabul) - egitim P99.99=6001.
- degisen_sayisi / boyali_sayisi: [0, 13] her biri (veri max: 12 / 13).
- degisen_sayisi + boyali_sayisi: <=13 (SERT kural) - egitim verisinde bu
  toplam HICBIR ZAMAN 13'u asmiyor; arabam.com'un 13 parcalik hasar
  tablosuyla (kaput, tavan, bagaj kapagi, on/arka tampon, 4 kapi, 4 camurluk)
  birebir eslesir - fiziksel olarak imkansiz bir kombinasyonu ifade eder.
- km_yil (kilometre/yas) > KM_YIL_WARNING_THRESHOLD: WARNING (ret degil) -
  P99.9=67.540 km/yil, ustu nadir ama egitim verisinde gozlemlenmis (max
  402.827) - "kesin yanlis" degil, sadece supheli.

DEGERLENDIRILIP KURAL EKLENMEYEN durumlar:
- Elektrikli + motor_hacmi>0: egitim verisindeki 320 elektrikli aracin
  cogunlugu zaten sifir olmayan motor_hacmi tasiyor (1300cc: 68 kez, 350cc:
  52 kez, ...) - veri kalitesi/semantik belirsizligi, bu NORM oldugu icin
  supheli olarak isaretlenmedi.
- agir_hasarli=1 + degisen_sayisi=0 + boyali_sayisi=0: egitim verisinin
  %15.6'sinda goruluyor (1192/7624 agir hasarli kayit) - yeterince yaygin,
  warning bile eklenmedi.
"""
from datetime import date

from preprocess import CURRENT_YEAR

# Takvim yili - preprocess.CURRENT_YEAR'dan bilerek ayri (bkz. modul docstring'i).
_CALENDAR_YEAR = date.today().year

FIELD_BOUNDS = {
    "yil": {"min": 1950, "max": _CALENDAR_YEAR + 1},
    "kilometre": {"min": 0, "max": 1_000_000},
    "motor_gucu": {"min": 1, "max": 2000},
    "motor_hacmi": {"min": 0, "max": 9000},
    "degisen_sayisi": {"min": 0, "max": 13},
    "boyali_sayisi": {"min": 0, "max": 13},
}

MAX_TOTAL_DAMAGED_PARTS = 13

# P99.9 (67.540) uzerine kucuk bir marj - tam esik degil, "nadir ama
# gozlemlenmis" ile "supheli" arasindaki gecis bandini biraz genisletir.
KM_YIL_WARNING_THRESHOLD = 70_000

REQUEST_FIELD_TO_CANONICAL = {
    "year": "yil",
    "mileage": "kilometre",
    "enginePower": "motor_gucu",
    "engineDisplacement": "motor_hacmi",
    "replacedPartsCount": "degisen_sayisi",
    "paintedPartsCount": "boyali_sayisi",
}

FIELD_LABELS = {
    "yil": "Yıl",
    "kilometre": "Kilometre",
    "motor_gucu": "Motor gücü",
    "motor_hacmi": "Motor hacmi",
    "degisen_sayisi": "Değişen parça sayısı",
    "boyali_sayisi": "Boyalı parça sayısı",
    "marka": "Marka",
    "model": "Model",
}


def _range_error(canonical_field, value):
    bounds = FIELD_BOUNDS[canonical_field]
    return {
        "code": "invalid_value",
        "field": canonical_field,
        "value": value,
        "message": f"{FIELD_LABELS[canonical_field]} kabul edilen aralığın dışında.",
        "allowed_range": bounds,
    }


def _blank_error(canonical_field, value):
    return {
        "code": "invalid_value",
        "field": canonical_field,
        "value": value,
        "message": f"{FIELD_LABELS[canonical_field]} boş veya yalnızca boşluk olamaz.",
        "allowed_range": None,
    }


def validate_domain(req) -> tuple[list, list]:
    """req: serve.PredictRequest. (errors, warnings) dondurur - errors
    doluysa cagiran 422 ile REDDETMELI; warnings basariya ragmen response'a
    eklenmeli (tahmini engellemez)."""
    errors = []
    warnings = []

    for request_field, canonical_field in REQUEST_FIELD_TO_CANONICAL.items():
        value = getattr(req, request_field)
        bounds = FIELD_BOUNDS[canonical_field]
        if value < bounds["min"] or value > bounds["max"]:
            errors.append(_range_error(canonical_field, value))

    # motor_hacmi: elektrik disinda 0/negatif kabul edilmez (FIELD_BOUNDS'daki
    # min=0 yalnizca ELEKTRIKLI icin gecerlidir - digerlerinde >0 sarti ayrica
    # kontrol edilir).
    if req.fuelType != "Elektrik" and req.engineDisplacement <= 0:
        errors.append({
            "code": "invalid_value",
            "field": "motor_hacmi",
            "value": req.engineDisplacement,
            "message": "Motor hacmi pozitif olmalı (elektrikli araç hariç).",
            "allowed_range": {"min": 1, "max": FIELD_BOUNDS["motor_hacmi"]["max"]},
        })

    brand_blank = not req.brand.strip()
    model_blank = not req.model.strip()
    if brand_blank:
        errors.append(_blank_error("marka", req.brand))
    if model_blank:
        errors.append(_blank_error("model", req.model))

    # marka/model bosken paket kabul edilmemeli (marka/model zaten zorunlu
    # oldugu icin normalde bu dal hicbir zaman tetiklenmez - savunma amacli).
    if req.trim.strip() and (brand_blank or model_blank):
        errors.append({
            "code": "inconsistent_combination",
            "field": "paket",
            "value": req.trim,
            "message": "Marka/model boşken paket kabul edilmez.",
            "allowed_range": None,
        })

    # degisen_sayisi + boyali_sayisi fiziksel parca sayisini asmamali (SERT
    # kural - egitim verisinde bu toplam hicbir zaman 13'u asmiyor).
    parts_total = req.replacedPartsCount + req.paintedPartsCount
    if parts_total > MAX_TOTAL_DAMAGED_PARTS:
        errors.append({
            "code": "inconsistent_combination",
            "field": "degisen_sayisi+boyali_sayisi",
            "value": parts_total,
            "message": (
                f"Değişen ({req.replacedPartsCount}) ve boyalı ({req.paintedPartsCount}) "
                f"parça toplamı ({parts_total}), bir aracın sahip olabileceği azami "
                f"parça sayısını ({MAX_TOTAL_DAMAGED_PARTS}) aşıyor."
            ),
            "allowed_range": {"max": MAX_TOTAL_DAMAGED_PARTS},
        })

    # yil+kilometre tutarliligi: asiri km/yil - WARNING (ret degil, egitim
    # verisinde nadir ama gozlemlenmis).
    if not errors:  # yil/kilometre zaten aralik disindaysa tekrar hesaplama
        yas = max(CURRENT_YEAR - req.year, 0)
        km_yil = req.mileage / yas if yas > 0 else req.mileage
        if km_yil > KM_YIL_WARNING_THRESHOLD:
            warnings.append({
                "code": "unusual_mileage_for_year",
                "message": (
                    f"Bu model yılı için kilometre değeri eğitim verisinde çok nadir "
                    f"({km_yil:,.0f} km/yıl, eğitim verisinin %99.9'u "
                    f"{KM_YIL_WARNING_THRESHOLD:,.0f} km/yıl altında)."
                ),
            })

    return errors, warnings
