"""Faz 7 Madde 5: normalize.py'yi arabam_test_val ve cars1'e uygulayip turetilmis
normalize edilmis kopyalarini uretir. Ham kaynaklar (data/output/arabam_test_val.csv -
scraper'in append hedefi, kaggle/cars1.csv - disaridan gelen orijinal veri seti)
degistirilmez; train_dataset.csv zaten prepare_train_dataset.py icinde normalize
edilerek uretiliyor (bkz. build_train_dataset).
"""
import os

import pandas as pd

from utils.normalize import normalize_dataframe

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def normalize_arabam_test_val():
    src = os.path.join(BASE_DIR, 'data', 'output', 'arabam_test_val.csv')
    dst = os.path.join(BASE_DIR, 'data', 'output', 'arabam_test_val_normalized.csv')
    df = pd.read_csv(src, low_memory=False, encoding='utf-8-sig')
    normalize_dataframe(df).to_csv(dst, index=False)
    print(f'{len(df)} kayit yazildi: {dst}')


def normalize_cars1():
    src = os.path.join(BASE_DIR, 'kaggle', 'cars1.csv')
    dst = os.path.join(BASE_DIR, 'data', 'output', 'cars1_normalized.csv')
    df = pd.read_csv(src, low_memory=False, encoding='utf-8-sig')
    normalize_dataframe(df, model_col='seri', kasa_col='kasa_tipi').to_csv(dst, index=False)
    print(f'{len(df)} kayit yazildi: {dst}')


def main():
    normalize_arabam_test_val()
    normalize_cars1()


if __name__ == '__main__':
    main()
