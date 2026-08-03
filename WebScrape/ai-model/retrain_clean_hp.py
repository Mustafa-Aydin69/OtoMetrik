"""Faz 17 (devam): peer-support guven mekanizmasi tamamlandiktan SONRA, Faz 16'da
probable_parse_error + physically_implausible olarak etiketlenen 140 kaydi
egitimden cikarip AYNI split/parametrelerle yeniden egitir - production
artefaktini GUNCELLER (models/lightgbm_final.joblib), hp_support da temizlenmis
veriden yeniden hesaplanir.

Karsilastirma ONCE/SONRA:
- negatif tahmin sayisi (dis holdout)
- en uc 20 tahmin (pozitif/negatif)
- 300+ HP segment bias/MAE
- Hyundai Accent 601 HP regresyon testi
- normal (<=200 HP) segmentlerde performans kaybi var mi

Calistirma (ai-model/ calisma dizini olarak): python retrain_clean_hp.py
"""
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hp_quality_analysis import attach_peer_stats, build_peer_stats, classify_hp_row
from train import (
    BASELINE_PARAMS, evaluate, load_model, prepare_external_holdout,
    prepare_full_training_data, save_model, smoke_test,
)


def compute_flagged_mask(X_full):
    primary, fallback = build_peer_stats(X_full)
    tagged = attach_peer_stats(X_full, primary, fallback)
    tagged['tag'] = tagged.apply(classify_hp_row, axis=1)
    return tagged['tag'].isin(['probable_parse_error', 'physically_implausible']).values


def segment_report(label, y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        print(f'{label}: n=0')
        return
    error = yp - yt
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f'{label}: n={len(yt):,} MAE={np.abs(error).mean():,.0f} RMSE={np.sqrt((error**2).mean()):,.0f} '
          f'bias={error.mean():+,.0f} R2={r2:.4f} negatif_tahmin={(yp<=0).sum()}')


def main():
    X_full, y_full = prepare_full_training_data()
    X_holdout, y_holdout = prepare_external_holdout(X_full)

    print('=== ONCE (mevcut production model, filtresiz egitim) ===')
    before_artifact = load_model()
    before_model = before_artifact['model']
    before_preds = before_model.predict(X_holdout)
    segment_report('GENEL', y_holdout.values, before_preds, np.ones(len(y_holdout), dtype=bool))
    hp = X_holdout['motor_gucu'].values
    segment_report('300+ HP', y_holdout.values, before_preds, hp >= 300)
    segment_report('<=200 HP (normal)', y_holdout.values, before_preds, hp <= 200)

    accent_mask = (X_holdout['marka'] == 'Hyundai') & (X_holdout['model'] == 'Accent') & (X_holdout['motor_gucu'] == 601)
    if accent_mask.any():
        idx = np.where(accent_mask.values)[0][0]
        print(f'Hyundai Accent 601 HP (ONCE): tahmin={before_preds[idx]:,.0f} TL '
              f'(gercek: {y_holdout.values[idx]:,.0f} TL)')

    top_pos_before = pd.Series(before_preds - y_holdout.values).nlargest(20)
    top_neg_before = pd.Series(before_preds - y_holdout.values).nsmallest(20)
    print(f'ONCE en uc 20 hata: pozitif ort={top_pos_before.mean():,.0f}, negatif ort={top_neg_before.mean():,.0f}')
    print()

    print('=== 140 supheli kayit cikariliyor ve yeniden egitiliyor ===')
    flagged_mask = compute_flagged_mask(X_full)
    print(f'cikarilan satir sayisi: {flagged_mask.sum()} / {len(X_full):,}')
    X_clean = X_full[~flagged_mask].reset_index(drop=True)
    y_clean = y_full[~flagged_mask].reset_index(drop=True)

    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(X_clean, y_clean)
    model_path = save_model(model, X_clean)
    print(f'yeni production model kaydedildi: {model_path}')
    print()

    print('=== SONRA (temizlenmis model) ===')
    after_preds = model.predict(X_holdout)
    segment_report('GENEL', y_holdout.values, after_preds, np.ones(len(y_holdout), dtype=bool))
    segment_report('300+ HP', y_holdout.values, after_preds, hp >= 300)
    segment_report('<=200 HP (normal)', y_holdout.values, after_preds, hp <= 200)

    if accent_mask.any():
        idx = np.where(accent_mask.values)[0][0]
        print(f'Hyundai Accent 601 HP (SONRA): tahmin={after_preds[idx]:,.0f} TL '
              f'(gercek: {y_holdout.values[idx]:,.0f} TL)')

    top_pos_after = pd.Series(after_preds - y_holdout.values).nlargest(20)
    top_neg_after = pd.Series(after_preds - y_holdout.values).nsmallest(20)
    print(f'SONRA en uc 20 hata: pozitif ort={top_pos_after.mean():,.0f}, negatif ort={top_neg_after.mean():,.0f}')

    print()
    print('=== smoke test (yeni artefakt) ===')
    smoke_test()


if __name__ == '__main__':
    main()
