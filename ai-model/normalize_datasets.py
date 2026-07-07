"""Faz 7 Madde 5: normalize.py'yi cars1'e uygulayip normalize edilmis kopyasini uretir.
Ham kaynak (kaggle/cars1.csv - disaridan gelen orijinal veri seti) degistirilmez.

Not: arabam_test_val.csv artik ayrica normalize edilmiyor - bu ham veri
prepare_train_dataset.py'nin load_arabam_scraped()'i uzerinden dogrudan
train_dataset.csv'ye karisip orada normalize ediliyor (bkz. build_train_dataset).
"""
import os

import pandas as pd

from utils.normalize import normalize_dataframe

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def normalize_cars1():
    src = os.path.join(BASE_DIR, 'kaggle', 'cars1.csv')
    dst = os.path.join(BASE_DIR, 'data', 'output', 'cars1_normalized.csv')
    df = pd.read_csv(src, low_memory=False, encoding='utf-8-sig')
    normalize_dataframe(df, model_col='seri', kasa_col='kasa_tipi').to_csv(dst, index=False)
    print(f'{len(df)} kayit yazildi: {dst}')


def main():
    normalize_cars1()


if __name__ == '__main__':
    main()
