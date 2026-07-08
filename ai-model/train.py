"""Faz 10 Madde 1: preprocess.py (Faz 9) ve baseline_gbm.py'de ayri ayri duran
kategori-tutarlilik mantigini (native kategori dtype, train/test kategori seti sabitleme)
tek bir yerde konsolide eden final on-isleme pipeline'i. baseline_gbm.py artik kendi
CATEGORICAL_COLS/to_category kopyasini tutmuyor, buradan iceri aktariyor.

Sonraki maddeler (hiperparametre ayari, final egitim, model+encoding serialize, smoke-test)
bu modul uzerine insa edilecek.
"""
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


# preprocess.load_clean_train_dataset() + split_features_target() + to_category() uclusunu
# tek cagrida toplar; hem baseline_gbm.py hem de bu dosyanin main()'i ayni hazirlanmis
# veriyi kullanir.
def prepare_training_data():
    df = load_clean_train_dataset()
    X_train, X_test, y_train, y_test = split_features_target(df)
    X_train_c, X_test_c = to_category(X_train, X_test)
    return X_train_c, X_test_c, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()
    print(f'train: {len(X_train)}, test: {len(X_test)}')
    print(X_train.dtypes.astype(str).to_string())


if __name__ == '__main__':
    main()
