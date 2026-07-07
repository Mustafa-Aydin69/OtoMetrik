"""Faz 9 Madde 1: Faz 8 EDA kararlarini uygulayan yeniden calistirilabilir on-isleme pipeline'i.

Uygulanan kararlar (bkz. eda_and_preprocessing.ipynb Madde 5):
- ust %1 fiyat + 1.000.000 km ustu kayitlar cikarilir (veri kirliligi).
- arac_turu / scraped_at cikarilir (train_dataset'te sirasiyla %100 / %95 eksik, sinyal tasimaz).
- degisen_sayisi / boyali_sayisi NaN'i "bilinmiyor" ayri flag kolonuyla isaretlenir, sonra 0'a doldurulur
  (dogrudan 0'a doldurmak "hasarsiz" gibi yanlis sinyal verirdi).
- yas = 2026 - yil, km_yil = kilometre / yas turetilir.
- Geri kalan kucuk oranli eksik degerler (marka/model/yakit_turu/vites/motor_hacmi/motor_gucu/yil)
  baseline asamasinda satir bazinda cikarilir.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_YEAR = 2026
DROP_COLS = ['arac_turu', 'scraped_at']
UNKNOWN_FLAG_COLS = ['degisen_sayisi', 'boyali_sayisi']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv')


def load_clean_train_dataset():
    df = pd.read_csv(TRAIN_PATH, low_memory=False)

    df = df[df['fiyat'] <= df['fiyat'].quantile(0.99)]
    df = df[df['kilometre'] <= 1_000_000]

    df = df.drop(columns=DROP_COLS)

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f'Kalan kucuk oranli eksik degerler icin {before - len(df)} satir cikarildi '
          f'({100 * (before - len(df)) / before:.2f}%)')

    return df


def split_features_target(df, test_size=0.2, random_state=42):
    y = df['fiyat']
    X = df.drop(columns=['fiyat', 'ilan_id'])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    df = load_clean_train_dataset()
    print(f'{len(df)} kayit (temizlenmis + turetilmis ozellikler dahil)')
    print(df.dtypes.astype(str).to_string())

    X_train, X_test, y_train, y_test = split_features_target(df)
    print(f'train: {len(X_train)}, test: {len(X_test)}')


if __name__ == '__main__':
    main()
