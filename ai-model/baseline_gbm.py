"""Faz 9 Madde 4: XGBoost / LightGBM baseline - native kategori destegiyle
(Faz 8 Madde 5 karari: agac tabanli modellerde one-hot/frekans yerine dogrudan kategori
destegi tercih edilmeli). String kolonlar 'category' dtype'a cevrilir, ikisi de bunu
kendi ic mekanizmasiyla isler - ayri encode() adimina gerek yok.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_clean_train_dataset, split_features_target

CATEGORICAL_COLS = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites']


# X_test'teki kategorileri X_train'de gorulenlerle sinirlar; train'de hic gorulmemis
# nadir degerler (orn. "Buick") NaN'a duser - XGBoost/LightGBM bunu native missing olarak isler.
def to_category(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category').cat.set_categories(X_train[col].cat.categories)
    return X_train, X_test


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}')
    return mae, rmse, r2


def main():
    df = load_clean_train_dataset()
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_c, X_test_c = to_category(X_train, X_test)

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
