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

Faz 10 Madde 3: final model artik train_dataset.csv'nin TAMAMIYLA (80/20 ic dogrulama split'i
degil) egitiliyor - Madde 2'de secilen baseline hiperparametreleriyle (tuning'de anlamli bir
kazanc bulunamadi). Ic split zaten yalnizca hiperparametre secimi icindi; secim netlestikten
sonra elimizdeki her satiri final modele vermek genelleme performansini artirir. Gercek
degerlendirme, egitimde hic gorulmemis dis holdout'ta (cars1_normalized.csv) yapilir.

Faz 10 Madde 4: model + encoding artefaktlari (kategori sutunlari, her sutunun train'de
gorulen kategori seti, ozellik sutun sirasi) tek bir joblib dosyasinda birlikte serialize
edilir - evaluate.py (Faz 11) egitim kodunu tekrar calistirmadan ayni donusumu uygulayabilsin.

Faz 10 Madde 5: kaydedilen artefakt geri yuklenip, egitimde hic kullanilmamis bir ornek
uzerinde smoke-test edilir (bkz. smoke_test()).
"""
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import CURRENT_YEAR, DROP_COLS, UNKNOWN_FLAG_COLS, load_clean_train_dataset, split_features_target

CATEGORICAL_COLS = ['marka', 'model', 'paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites']

# Madde 2 (tune_lightgbm.py) sonucu: genisletilmis arama platoyu asamadi, varsayilan
# hiperparametreler korunuyor.
BASELINE_PARAMS = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1, verbose=-1)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CARS1_HOLDOUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'cars1_normalized.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_final.joblib')

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


# Ic 80/20 split olmadan, train_dataset.csv'nin tamamini final model icin hazirlar - satir
# sayisini artirmak genelleme performansini artirir, ic split zaten sadece Madde 2'nin
# hiperparametre secimi icindi.
def prepare_full_training_data():
    df = load_clean_train_dataset()
    y = df['fiyat']
    X = df.drop(columns=['fiyat', 'ilan_id'])
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype('category')
    return X, y


def train_final_model():
    X, y = prepare_full_training_data()
    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(X, y)
    return model, X, y


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.4f}')
    return mae, rmse, r2


# Model + encoding artefaktlarini (kategori sutunlari, train'de gorulen kategori seti,
# ozellik sutun sirasi) tek dosyada birlikte kaydeder - evaluate.py (Faz 11) train.py'yi
# tekrar calistirmadan ayni donusumu apply_saved_categories() ile uygulayabilir.
def save_model(model, X_full):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        'model': model,
        'categorical_cols': CATEGORICAL_COLS,
        'category_sets': {col: X_full[col].cat.categories for col in CATEGORICAL_COLS},
        'feature_columns': list(X_full.columns),
    }
    joblib.dump(artifact, MODEL_PATH)
    return MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


# Kaydedilen artefaktla, egitimde kullanilan X_full nesnesine erisimi olmayan bir cagiran
# (orn. evaluate.py) icin ham bir X'i modelin bekledigi sutun sirasina/kategori setine
# hizalar.
def apply_saved_categories(X, artifact):
    X = X.reindex(columns=artifact['feature_columns']).copy()
    for col in artifact['categorical_cols']:
        X[col] = X[col].astype('category').cat.set_categories(artifact['category_sets'][col])
    return X


# Faz 10 Madde 5: diskten geri yuklenen artefaktin, egitimde hic kullanilmamis dis holdout'un
# ilk kaydinda bellek-ici modelle ayni tahmini urettigini dogrular.
def smoke_test():
    artifact = load_model()
    df = load_cars1_holdout()
    sample = df.drop(columns=['fiyat', 'ilan_id']).iloc[[0]]
    sample_aligned = apply_saved_categories(sample, artifact)
    pred = artifact['model'].predict(sample_aligned)[0]
    actual = df['fiyat'].iloc[0]
    print(f'smoke-test -> yeniden yuklenen model tahmini: {pred:,.0f} (gercek fiyat: {actual:,.0f})')
    return pred


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()
    print(f'ic dogrulama -> train: {len(X_train)}, test: {len(X_test)}')
    print(X_train.dtypes.astype(str).to_string())

    X_holdout, y_holdout = prepare_external_holdout(X_train)
    print(f'\ndis holdout (cars1_normalized.csv) -> {len(X_holdout)} kayit')
    print(f'  renk bilinmiyor: %{100 * X_holdout["renk"].isna().mean():.1f} (cars1de renk alani yok)')
    print(f'  agir_hasarli=0 varsayilan: %{100 * (X_holdout["agir_hasarli"] == 0).mean():.1f} '
          f'(cars1de agir_hasarli alani yok)')

    print('\n--- final model (train_dataset.csv tamami + baseline hiperparametreleri) ---')
    model, X_full, y_full = train_final_model()
    print(f'egitim seti: {len(X_full)} kayit (ic 80/20 split degil, tum temiz veri)')
    evaluate(y_full, model.predict(X_full), 'train (tam veri, ic gorulmus)')

    X_holdout_full, y_holdout_full = prepare_external_holdout(X_full)
    evaluate(y_holdout_full, model.predict(X_holdout_full), 'dis holdout (cars1, hic gorulmemis)')

    model_path = save_model(model, X_full)
    print(f'\nmodel kaydedildi: {model_path}')
    smoke_test()


if __name__ == '__main__':
    main()
