"""Faz 9 Madde 4: XGBoost / LightGBM baseline - native kategori destegiyle
(Faz 8 Madde 5 karari: agac tabanli modellerde one-hot/frekans yerine dogrudan kategori
destegi tercih edilmeli). String kolonlar 'category' dtype'a cevrilir, ikisi de bunu
kendi ic mekanizmasiyla isler - ayri encode() adimina gerek yok.

Kategori-donusum mantigi Faz 10 Madde 1'de train.py'ye tasindi (CATEGORICAL_COLS,
to_category, prepare_training_data); bu dosya artik oradan iceri aktariyor.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from train import CATEGORICAL_COLS, prepare_training_data


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}')
    return mae, rmse, r2


def main():
    X_train_c, X_test_c, y_train, y_test = prepare_training_data()

    print('--- XGBoost ---')
    xgb = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                        enable_categorical=True, random_state=42, n_jobs=-1)
    xgb.fit(X_train_c, y_train)
    evaluate(y_train, xgb.predict(X_train_c), 'train')
    evaluate(y_test, xgb.predict(X_test_c), 'test')

    print('--- LightGBM ---')
    lgbm = LGBMRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                          random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(X_train_c, y_train, categorical_feature=CATEGORICAL_COLS)
    evaluate(y_train, lgbm.predict(X_train_c), 'train')
    evaluate(y_test, lgbm.predict(X_test_c), 'test')


if __name__ == '__main__':
    main()
