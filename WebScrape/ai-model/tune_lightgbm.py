"""Faz 10 Madde 3: LightGBM hiperparametre ayari - learning_rate/n_estimators/num_leaves/
max_depth uzerinde arama yaparak mevcut baseline_gbm.py sonucunu (ic dogrulama test
R2=0.9668) gecmeyi hedefler.

Ilk deneme (sadece bu 4 parametre, scoring=neg_MAE) num_leaves=127/max_depth=-1/n_estimators=1000
gibi kisitsiz bir agac secti: MAE'de kil payi iyilesme (63.155->62.640) verdi ama R2'yi
dusurdu (0.9668->0.9638) ve train/test farkini buyuterek asiri ogrenmeyi artirdi. Bu 4 parametre
regularizasyon icermedigi icin genis num_leaves/derin agac kombinasyonlari cezalandirilmiyordu.
Bu yuzden arama alanina regularizasyon parametreleri (min_child_samples, reg_alpha, reg_lambda)
eklendi, max_depth'ten sinirsiz (-1) secenegi cikarildi, scoring MAE yerine r2'ye cevrildi
(hedef metrikle dogrudan hizalanmasi icin).

Secim SADECE train_dataset.csv icindeki ic dogrulama (train/test) split'i uzerinden yapilir -
train.prepare_external_holdout()'un urettigi cars1 dis holdout'u kasten KULLANILMAZ; aksi halde
hiperparametreleri dis holdout'a gore secmis oluruz ve Faz 11'in gercek genelleme testi olma
amacini bastan kirleriz (tuning-to-testset leakage). En iyi konfigurasyon bulunduktan sonra
sadece bilgi amacli dis holdout'ta da bir kez raporlanir, ama secim buna gore yapilmaz.
"""
import time

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from train import prepare_external_holdout, prepare_training_data

PARAM_DISTRIBUTIONS = {
    'n_estimators': [300, 400, 600, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.08],
    'num_leaves': [15, 31, 63, 95],
    'max_depth': [4, 6, 8, 10, 12],
    'min_child_samples': [10, 20, 30, 50, 100],
    'reg_alpha': [0, 0.1, 0.5, 1, 5],
    'reg_lambda': [0, 0.1, 0.5, 1, 5],
}
N_ITER = 40
CV_FOLDS = 3


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}')
    return mae, rmse, r2


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()

    search = RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='r2',
        random_state=42,
        n_jobs=1,
    )

    print(f'{N_ITER} konfigurasyon x {CV_FOLDS}-fold CV taraniyor (ic dogrulama train split uzerinde)...')
    start = time.time()
    search.fit(X_train, y_train)
    print(f'Tarama {time.time() - start:.0f}s surdu.')

    print(f'\nEn iyi hiperparametreler: {search.best_params_}')
    print(f'En iyi CV R2: {search.best_score_:.4f}')

    best_model = search.best_estimator_
    print('\n--- Baseline (varsayilan hiperparametreler, karsilastirma icin) ---')
    baseline = LGBMRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                              random_state=42, n_jobs=-1, verbose=-1)
    baseline.fit(X_train, y_train)
    evaluate(y_test, baseline.predict(X_test), 'test (baseline)')

    print('\n--- Ayarlanmis LightGBM ---')
    evaluate(y_train, best_model.predict(X_train), 'train')
    evaluate(y_test, best_model.predict(X_test), 'test')

    print('\n--- Bilgi amacli: dis holdout (cars1_normalized.csv) - SECIME dahil edilmedi ---')
    X_holdout, y_holdout = prepare_external_holdout(X_train)
    evaluate(y_holdout, best_model.predict(X_holdout), 'cars1 holdout')


if __name__ == '__main__':
    main()
