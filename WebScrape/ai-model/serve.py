"""Faz 12: train.py (Faz 10) tarafindan kaydedilen LightGBM artefaktini
FastAPI uzerinden /predict olarak yayinlar. WebSite/src/app/api/predict/route.ts
buraya PREDICTION_API_URL ile istek atar; yanit sozlesmesi (price alani) o
route'un beklentisiyle birebir eslesir.

Calistirma (ai-model/ calisma dizini olarak):
    uvicorn serve:app --host 0.0.0.0 --port 8000

Once models/lightgbm_final.joblib'in var olmasi gerekir (bkz. train.py).
"""
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


# Egitim zamaninda preprocess.load_clean_train_dataset()'in uyguladigi
# turetme/eslemeyi tek bir istek satiri icin yeniden uretir; sutun sirasi ve
# kategori setleri apply_saved_categories() ile artefakttaki egitim zamani
# degerlerine sabitlenir (train'de gorulmemis degerler native missing olur).
def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    yas = max(CURRENT_YEAR - req.year, 0)
    km_yil = req.mileage / (yas if yas > 0 else 1)
    trim = req.trim.strip() or np.nan

    return pd.DataFrame([{
        "marka": req.brand,
        "model": req.model,
        "paket": trim,
        "kasa_turu": req.bodyType,
        "renk": req.color,
        "motor_hacmi": req.engineDisplacement,
        "motor_gucu": req.enginePower,
        "yil": req.year,
        "kilometre": req.mileage,
        "yakit_turu": req.fuelType,
        "vites": req.transmission,
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
    return {"status": "ok", "model_loaded": MODEL_ARTIFACT is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    row = build_feature_row(req)
    X = apply_saved_categories(row, MODEL_ARTIFACT)
    pred = float(MODEL_ARTIFACT["model"].predict(X)[0])

    if not np.isfinite(pred) or pred <= 0:
        raise HTTPException(status_code=502, detail="Model gecerli bir tahmin uretmedi.")

    return {"price": round(pred), "currency": "TRY", "source": "model"}
