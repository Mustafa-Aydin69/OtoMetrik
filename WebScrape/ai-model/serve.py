"""Faz 12: train.py (Faz 10) tarafindan kaydedilen LightGBM artefaktini
FastAPI uzerinden /predict olarak yayinlar. WebSite/src/app/api/predict/route.ts
buraya PREDICTION_API_URL ile istek atar; yanit sozlesmesi (price alani) o
route'un beklentisiyle birebir eslesir.

Faz 13: kategorik alanlar (marka/vites/yakit_turu/kasa_turu/renk) icin
category_mapping.py'deki TEK kaynaga karsi dogrulama yapilir. Website'in
kullanici etiketini (orn. "Manuel") DOGRUDAN gonderdigi durumlar icin de
(savunma katmani - route.ts kanonik ceviriyi atlarsa/eskiyse) resolve_canonical()
etiketi kanonik degere (orn. "Düz") cevirir; ne etiket ne kanonik olarak
taninan bir deger 422 ile REDDEDILIR (sessizce NaN'a dusurulmez, bkz.
dagitima hazirlik analizindeki "Manuel/Düz" ve "Mercedes-Benz/Mercedes - Benz"
bulgulari).

Faz 14: domain_validation.py'deki egitim-verisi-turevli alan sinirlari ve
alanlar-arasi tutarlilik kurallari da ayni 422 akisina eklenir. Tum
dogrulama hatalari (kategori + domain) TEK bir gecist e toplanir; birden
fazla hata varsa detail bir LISTE, tek hata varsa detail tek bir OBJE olur
(her ikisinde de ayni {code, field, value, message, allowed_range|
allowed_examples} sekli). Kesin-yanlis olmayan ama supheli durumlar (orn.
asiri km/yil) tahmini ENGELLEMEZ - basarili yanitin "warnings" alanina eklenir.

Calistirma (ai-model/ calisma dizini olarak):
    uvicorn serve:app --host 0.0.0.0 --port 8000

Once models/lightgbm_final.joblib'in var olmasi gerekir (bkz. train.py).
"""
from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from category_mapping import LABEL_TO_CANONICAL, UNSUPPORTED_LABELS, allowed_examples, resolve_canonical
from domain_validation import validate_domain
from preprocess import CURRENT_YEAR
from train import apply_saved_categories, load_model

app = FastAPI(title="OtoMetrik Fiyat Tahmin Servisi")

try:
    MODEL_ARTIFACT = load_model()
except FileNotFoundError as exc:
    raise RuntimeError(
        "Model artefakti bulunamadi (models/lightgbm_final.joblib). "
        "Once 'python train.py' calistirin."
    ) from exc

# "yas"/"km_yil" ozelliklerinin egitimde hangi takvim yiliyla hesaplandigi -
# artefaktan okunur (bkz. train.py save_model() yorumu). Eski (bu alani
# icermeyen) artefaktlarla geriye donuk uyumluluk icin preprocess.CURRENT_YEAR'a
# duser, ama normal calismada HER ZAMAN artefaktin kendi degeri kullanilir -
# takvim yili ilerledikce bu deger SESSIZCE degistirilmez (bkz. modul
# docstring'i, Faz 16).
MODEL_REFERENCE_YEAR = MODEL_ARTIFACT.get('reference_year', CURRENT_YEAR)

# {"marka": [...], "vites": [...], ...} - resolve_canonical() ve 422 hata
# mesajlarindaki allowed_examples icin bir kez hesaplanir.
CATEGORY_SETS = {col: list(cats) for col, cats in MODEL_ARTIFACT['category_sets'].items()}

# API alan adi (PredictRequest) -> kanonik sema alan adi (category_mapping.py/egitim).
REQUEST_FIELD_TO_CANONICAL_FIELD = {
    'brand': 'marka',
    'transmission': 'vites',
    'fuelType': 'yakit_turu',
    'bodyType': 'kasa_turu',
    'color': 'renk',
}


# WebSite/src/lib/validation.ts'teki PredictionInput ile birebir ayni sozlesme.
class PredictRequest(BaseModel):
    brand: str
    model: str
    year: int
    mileage: float
    fuelType: str
    transmission: str
    bodyType: str
    color: str
    engineDisplacement: float
    enginePower: float
    trim: str = ""
    replacedPartsCount: int
    paintedPartsCount: int
    heavyDamage: bool


# req'teki 5 kategorik alani (marka/vites/yakit_turu/kasa_turu/renk) kanonik
# degerlere cevirir. Kullanici etiketi (orn. "Manuel") VEYA zaten kanonik bir
# deger (orn. "Düz") kabul edilir; ikisi de degilse hata listesine eklenir -
# apply_saved_categories()'in unseen-category'yi sessizce NaN'a dusurmesine
# hic firsat verilmez (bkz. modul docstring'i, Faz 13). Faz 14: artik hemen
# raise ETMEZ - domain hatalarıyla BIRLIKTE tek bir 422 yanitinda toplanabilsin
# diye (errors, resolved) dondurur.
def collect_category_errors(req: "PredictRequest") -> tuple[list, dict]:
    errors = []
    resolved = {}
    for request_field, canonical_field in REQUEST_FIELD_TO_CANONICAL_FIELD.items():
        raw_value = getattr(req, request_field)
        # marka bos/whitespace ise domain_validation.validate_domain() bunu zaten
        # ayri (ve daha net: "bos deger") bir hata olarak raporlar - burada
        # "bilinmeyen kategori" olarak TEKRAR eklenip ayni kok nedenin iki kez
        # raporlanmasi (gurultu) onlenir.
        if canonical_field == "marka" and not raw_value.strip():
            continue
        ok, canonical_value = resolve_canonical(canonical_field, raw_value, CATEGORY_SETS)
        if not ok:
            errors.append({
                "code": "unknown_category",
                "field": canonical_field,
                "value": raw_value,
                "message": "Bilinmeyen kategori.",
                "allowed_examples": allowed_examples(canonical_field, CATEGORY_SETS),
            })
        else:
            resolved[canonical_field] = canonical_value
    return errors, resolved


# Egitim zamaninda preprocess.load_clean_train_dataset()'in uyguladigi
# turetme/eslemeyi tek bir istek satiri icin yeniden uretir; sutun sirasi ve
# kategori setleri apply_saved_categories() ile artefakttaki egitim zamani
# degerlerine sabitlenir (train'de gorulmemis degerler native missing olur).
def build_feature_row(req: PredictRequest, resolved_categories: dict) -> pd.DataFrame:
    yas = max(MODEL_REFERENCE_YEAR - req.year, 0)
    km_yil = req.mileage / (yas if yas > 0 else 1)
    trim = req.trim.strip() or np.nan

    return pd.DataFrame([{
        "marka": resolved_categories["marka"] or np.nan,
        "model": req.model,
        "paket": trim,
        "kasa_turu": resolved_categories["kasa_turu"],
        "renk": resolved_categories["renk"],
        "motor_hacmi": req.engineDisplacement,
        "motor_gucu": req.enginePower,
        "yil": req.year,
        "kilometre": req.mileage,
        "yakit_turu": resolved_categories["yakit_turu"],
        "vites": resolved_categories["vites"],
        "degisen_sayisi": req.replacedPartsCount,
        "boyali_sayisi": req.paintedPartsCount,
        "agir_hasarli": int(req.heavyDamage),
        "degisen_sayisi_bilinmiyor": 0,
        "boyali_sayisi_bilinmiyor": 0,
        "yas": yas,
        "km_yil": km_yil,
    }])


@app.get("/health")
def health():
    """model_reference_year: "yas" ozelliginin egitimde hangi takvim yiliyla
    hesaplandigi (bkz. train.py save_model() / modul ici not). model_age_years,
    takvim yili bu referanstan ne kadar ilerlemis - buyudukce "yas" ozelligi
    serve-time'da giderek daha yanlis hesaplanir (yeni araclar olduklarindan
    daha yasli gorunur); operasyonel bir staleness sinyali olarak izlenmelidir -
    esik asilinca cozum yeniden egitimdir, serve-time formulu degistirmek degil."""
    return {
        "status": "ok",
        "model_loaded": MODEL_ARTIFACT is not None,
        "model_reference_year": MODEL_REFERENCE_YEAR,
        "model_age_years": date.today().year - MODEL_REFERENCE_YEAR,
    }


@app.get("/categories")
def categories():
    """Frontend'in dropdown secenekleri icin kullanabilecegi TEK dogruluk
    kaynagi (bkz. category_mapping.py). Website build-time'da uretilen
    category-options.generated.ts'i kullanir; bu endpoint aynı veriyi
    calisma-zamaninda dogrulamak/entegre etmek isteyenler icindir."""
    return {
        field: {
            "options": [{"label": label, "value": canonical} for label, canonical in labels.items()],
            "unsupported": UNSUPPORTED_LABELS.get(field, {}),
        }
        for field, labels in LABEL_TO_CANONICAL.items()
    }


@app.post("/predict")
def predict(req: PredictRequest):
    category_errors, resolved_categories = collect_category_errors(req)
    domain_errors, warnings = validate_domain(req)

    all_errors = category_errors + domain_errors
    if all_errors:
        raise HTTPException(status_code=422, detail=all_errors[0] if len(all_errors) == 1 else all_errors)

    row = build_feature_row(req, resolved_categories)
    X = apply_saved_categories(row, MODEL_ARTIFACT)
    pred = float(MODEL_ARTIFACT["model"].predict(X)[0])

    if not np.isfinite(pred) or pred <= 0:
        raise HTTPException(status_code=502, detail="Model gecerli bir tahmin uretmedi.")

    return {"price": round(pred), "currency": "TRY", "source": "model", "warnings": warnings}
