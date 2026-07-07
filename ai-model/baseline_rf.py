"""Faz 9 Madde 3: RandomForest baseline - Linear Regression referansiyla karsilastirma.

Agac tabanli oldugu icin olceklemeye ihtiyac duymaz; ayni encode() (preprocess.py) kullanilir.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train_enc, y_train)

    evaluate(y_train, model.predict(X_train_enc), 'train')
    evaluate(y_test, model.predict(X_test_enc), 'test')

    importances = pd.Series(model.feature_importances_, index=X_train_enc.columns).sort_values(ascending=False)
    print('En onemli 10 ozellik:')
    print(importances.head(10).round(4).to_string())


if __name__ == '__main__':
    main()
