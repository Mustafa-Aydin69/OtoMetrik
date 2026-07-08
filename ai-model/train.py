"""Faz 10 Madde 1: preprocess.py (Faz 9) ve baseline_gbm.py'de ayri ayri duran
kategori-tutarlilik mantigini (native kategori dtype, train/test kategori seti sabitleme)
tek bir yerde konsolide eden final on-isleme pipeline'i. baseline_gbm.py artik kendi
CATEGORICAL_COLS/to_category kopyasini tutmuyor, buradan iceri aktariyor.

Faz 10 Madde 2: holdout stratejisi netlestirildi. train_dataset.csv icindeki 80/20 split
(prepare_training_data) sadece IC dogrulama icindir (hiperparametre ayari, Madde 3) - bu
veri kaynagimizin kendi kazidigimiz arabam_test_val.csv'yi de icerdigi icin gercek bir
"gormedigi veri" testi degil. Gercek/dis genelleme testi kaggle/cars1.csv'den turetilen
data/output/cars1_normalized.csv'dir (bkz. roadmap Faz 5 karari) - bu modul + evaluate.py
(Faz 11) o veriyi kullanacak.

Sonraki maddeler (hiperparametre ayari, final egitim, model+encoding serialize, smoke-test)
bu modul uzerine insa edilecek.
"""
import os

import numpy as np
import pandas as pd

from preprocess import CURRENT_YEAR, DROP_COLS, UNKNOWN_FLAG_COLS, load_clean_train_dataset, split_features_target

CATEGORICAL_COLS = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CARS1_HOLDOUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'cars1_normalized.csv')

# cars1_normalized.csv kendi ham kaggle/cars1.csv sutun adlarini korur (normalize_datasets.py
# sadece degerleri normalize eder, semaya map etmez) - train_dataset.csv'nin kanonik semasina
# eslemek icin burada yeniden adlandiriyoruz.
CARS1_COLUMN_MAP = {
    'seri': 'model',
    'model': 'paket',
    'vites_tipi': 'vites',
    'yakit_tipi': 'yakit_turu',
    'kasa_tipi': 'kasa_turu',
    'degisen': 'degisen_sayisi',
    'boyali': 'boyali_sayisi',
}
# cars1'de kanonik semadaki iki alanin karsiligi yok: renk hic yok, agir_hasarli hic yok
# (tramer TL tazminat tutari farkli bir kavram, dogrudan karsiligi degil). "bilinmiyor"
# olarak eklenip digerleriyle (degisen_sayisi/boyali_sayisi) ayni konvansiyonla islenir.
CARS1_MISSING_COLS = ['renk', 'agir_hasarli']
# Kanonik semada karsiligi olmayan cars1'e ozgu alanlar - model bunlari kullanmiyor.
CARS1_EXTRA_COLS = ['konum', 'cekis', 'ortalama_yakit_tuketimi', 'yakit_deposu', 'tramer']


# X_test'teki kategorileri X_train'de gorulenlerle sinirlar; train'de hic gorulmemis
# nadir degerler (orn. "Buick") NaN'a duser - XGBoost/LightGBM bunu native missing olarak isler.
def to_category(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category').cat.set_categories(X_train[col].cat.categories)
    return X_train, X_test


# preprocess.load_clean_train_dataset() + split_features_target() + to_category() uclusunu
# tek cagrida toplar; hem baseline_gbm.py hem de bu dosyanin main()'i ayni hazirlanmis
# veriyi kullanir.
def prepare_training_data():
    df = load_clean_train_dataset()
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_c, X_test_c = to_category(X_train, X_test)
    return X_train_c, X_test_c, y_train, y_test


# cars1_normalized.csv'yi kanonik semaya tasiyip preprocess.py'nin ayni temizleme adimlarini
# (ust %1 fiyat + 1M km filtre, yas/km_yil turetme) uygular. Tek fark: preprocess.py'nin son
# adimindaki genel dropna() burada UYGULANMAZ - renk/agir_hasarli cars1'de %100 eksik oldugu
# icin o adim tum satirlari silerdi. Onun yerine ikisi de digerleriyle (degisen_sayisi/
# boyali_sayisi) ayni "bilinmiyor" konvansiyonuyla doldurulur, sadece renk gercekten NaN kalir
# (frekans/kategori kodlamasi bunu zaten native olarak "gorulmemis deger" gibi isler).
def load_cars1_holdout():
    df = pd.read_csv(CARS1_HOLDOUT_PATH, low_memory=False, encoding='utf-8-sig')
    df = df.rename(columns=CARS1_COLUMN_MAP).drop(columns=CARS1_EXTRA_COLS, errors='ignore')
    for col in CARS1_MISSING_COLS:
        df[col] = np.nan
    df['ilan_id'] = 'cars1-' + df.index.astype(str)

    df = df[df['fiyat'] <= df['fiyat'].quantile(0.99)]
    df = df[df['kilometre'] <= 1_000_000]
    df = df.drop(columns=DROP_COLS, errors='ignore')

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)
    df['agir_hasarli'] = df['agir_hasarli'].fillna(0)
    df['renk_bilinmiyor'] = df['renk'].isna().astype(int)

    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)

    non_renk_cols = [c for c in df.columns if c != 'renk']
    return df.dropna(subset=non_renk_cols).reset_index(drop=True)


# Dis holdout'u X_train ile ayni kolon sirasina/kategori setine sabitler (to_category'deki
# "train'de gorulmemis -> NaN" mantigi burada da uygulanir), boylece model dogrudan predict
# edebilir.
def prepare_external_holdout(X_train):
    df = load_cars1_holdout()
    y_holdout = df['fiyat']
    X_holdout = df.drop(columns=['fiyat', 'ilan_id']).reindex(columns=X_train.columns)
    for col in CATEGORICAL_COLS:
        X_holdout[col] = X_holdout[col].astype('category').cat.set_categories(X_train[col].cat.categories)
    return X_holdout, y_holdout


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()
    print(f'ic dogrulama -> train: {len(X_train)}, test: {len(X_test)}')
    print(X_train.dtypes.astype(str).to_string())

    X_holdout, y_holdout = prepare_external_holdout(X_train)
    print(f'\ndis holdout (cars1_normalized.csv) -> {len(X_holdout)} kayit')
    print(f'  renk bilinmiyor: %{100 * X_holdout["renk"].isna().mean():.1f} (cars1de renk alani yok)')
    print(f'  agir_hasarli=0 varsayilan: %{100 * (X_holdout["agir_hasarli"] == 0).mean():.1f} '
          f'(cars1de agir_hasarli alani yok)')


if __name__ == '__main__':
    main()
