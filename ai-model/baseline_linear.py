"""Faz 9 Madde 2: Linear Regression baseline - referans MAE/RMSE/R2."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from preprocess import encode, load_clean_train_dataset, split_features_target


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
