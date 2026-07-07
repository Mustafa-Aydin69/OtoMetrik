"""Faz 9 Madde 2: Linear Regression baseline - referans MAE/RMSE/R2.

Kategorik encoding: dusuk kardinaliteli alanlar (yakit_turu, vites, kasa_turu) one-hot;
yuksek kardinaliteli alanlar (marka, model, paket, renk) frekans encoding (X_train'e gore
fit edilir, sizinti olmasin diye X_test'e sadece transform uygulanir).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from preprocess import load_clean_train_dataset, split_features_target

ONEHOT_COLS = ['yakit_turu', 'vites', 'kasa_turu']
FREQ_COLS = ['marka', 'model', 'paket', 'renk']


def encode(X_train, X_test):
    freq_maps = {col: X_train[col].value_counts() for col in FREQ_COLS}
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in FREQ_COLS:
        X_train_enc[col] = X_train[col].map(freq_maps[col]).fillna(0)
        X_test_enc[col] = X_test[col].map(freq_maps[col]).fillna(0)

    X_train_enc = pd.get_dummies(X_train_enc, columns=ONEHOT_COLS)
    X_test_enc = pd.get_dummies(X_test_enc, columns=ONEHOT_COLS)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc, X_test_enc


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}')
    return mae, rmse, r2


def main():
    df = load_clean_train_dataset()
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_enc, X_test_enc = encode(X_train, X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    evaluate(y_train, model.predict(X_train_scaled), 'train')
    evaluate(y_test, model.predict(X_test_scaled), 'test')


if __name__ == '__main__':
    main()
