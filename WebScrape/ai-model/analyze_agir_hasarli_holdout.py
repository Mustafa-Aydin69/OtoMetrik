"""Faz 11 Madde 6: agir_hasarli icin stratified, gercek out-of-sample holdout testi.

Motivasyon: Faz 11 Madde 2'deki segment kontrolu in-sample'di (final model tum
train_dataset.csv ile egitildigi icin gercek "gormedigi veri" degildi) ve dis holdout
(cars1_normalized.csv) bu alani hic icermiyor. Bu modul, agir_hasarli'nin dusuk global
SHAP/importance payinin nedenini ayirt eder: model segmenti ogrenemiyor mu (yuksek MAE),
sistematik yanli mi (pozitif bias), yoksa segment performansi normal de etkisi diger
degiskenler tarafindan mi emiliyor?

Kritik veri kalitesi bulgusu: arabam_test_val.csv'deki agir_hasarli, iki farkli "bilinmiyor"
durumunu ayirt etmiyordu. src/utils/json-parser.js'deki
`agir_hasarli: raw.agirHasarli !== undefined ? parseEvetHayir(raw.agirHasarli) : 0` satiri,
alan ilan detay tablosunda bulunamadiginda sessizce 0 (hasarsiz) yaziyor. train_dataset.csv'de
bu, arabam-kaynakli satirlarin bir kisminda GERCEK NaN (eski scrape donemi, alan henuz
DETAIL_LABEL_MAP'e eklenmemisti - bkz. preprocess.py "eski canli kazima kayitlari" yorumu),
digerlerinde ise 0/1 (alan takip edilmeye basladiktan sonra) olarak goruluyor. preprocess.py
`fillna(0)` ile ikisini de "hasarsiz" sayiyor - bu, ~53k satiri (train_dataset.csv'nin
%18'i) yanlislikla hasarsiz etiketliyor. Bu modul SADECE gercekten etiketlenmis (NaN
olmayan) arabam satirlarini kullanir; bu spesifik yanlis-etiketleme riskini ortadan
kaldirmaz (ilan bazinda hala "alan yok" ile "hasarsiz" karisabilir, bkz. rapor sonundaki not)
ama en azindan "hic scrape edilmemis donem" kaynakli sistematik hatayi eler.

Split: arabam satirlari (gercek etiketli) uzerinde train/val/test (%70/%15/%15),
stratify=agir_hasarli, random_state sabit. Model SADECE train ile egitilir; test seti
egitimde hic gorulmez. val seti bu turda kullanilmiyor (erken durdurma/hiperparametre
ayari yapilmiyor, production BASELINE_PARAMS ile ayni ayarlar kullaniliyor) - sadece
split'in ayrildigini kayit altina almak icin raporlanir.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from preprocess import CURRENT_YEAR, DROP_COLS, UNKNOWN_FLAG_COLS
from train import BASELINE_PARAMS, CATEGORICAL_COLS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv')
RANDOM_STATE = 42


# preprocess.load_clean_train_dataset() ile ayni temizleme adimlari (ust %1 fiyat + 1M km
# filtre, degisen/boyali unknown-flag+fill0, yas/km_yil turetme) - TEK fark: agir_hasarli
# fillna(0) BURADA UYGULANMAZ, cunku bu modulun butun amaci gercek etiketle sahte etiketi
# ayirmak. Sadece arabam-kaynakli (ilan_id 'arabam-' ile baslayan) VE agir_hasarli NaN
# OLMAYAN satirlar tutulur.
def load_labeled_arabam_subset():
    df = pd.read_csv(TRAIN_PATH, low_memory=False, encoding='utf-8-sig')
    df = df[df['ilan_id'].astype(str).str.startswith('arabam-')]
    df = df[df['agir_hasarli'].notna()]

    df = df[df['fiyat'] <= df['fiyat'].quantile(0.99)]
    df = df[df['kilometre'] <= 1_000_000]
    df = df.drop(columns=DROP_COLS, errors='ignore')

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f'kucuk oranli eksik degerler icin {before - len(df)} satir cikarildi '
          f'({100 * (before - len(df)) / before:.2f}%)')
    return df


def stratified_split(df):
    y = df['fiyat']
    X = df.drop(columns=['fiyat', 'ilan_id'])
    strat = df['agir_hasarli']

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, strat, test_size=0.30, stratify=strat, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, stratify=s_temp, random_state=RANDOM_STATE)

    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype('category')
        for X_other in (X_val, X_test):
            X_other[col] = X_other[col].astype('category').cat.set_categories(X_train[col].cat.categories)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def segment_report(X_test, y_test, preds):
    rows = []
    for value, label in [(0, 'Hayır'), (1, 'Evet')]:
        mask = (X_test['agir_hasarli'] == value).values
        n = mask.sum()
        if n == 0:
            continue
        y_seg = y_test.values[mask]
        p_seg = preds[mask]
        mae = np.mean(np.abs(y_seg - p_seg))
        rmse = np.sqrt(np.mean((y_seg - p_seg) ** 2))
        ss_res = np.sum((y_seg - p_seg) ** 2)
        ss_tot = np.sum((y_seg - y_seg.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rows.append({
            'segment': f'Ağır hasarlı = {label}',
            'n': n,
            'gercek_ortalama_fiyat': y_seg.mean(),
            'tahmin_ortalamasi': p_seg.mean(),
            'bias': p_seg.mean() - y_seg.mean(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        })
    return pd.DataFrame(rows)


def main():
    df = load_labeled_arabam_subset()
    print(f'gercek etiketli arabam satiri: {len(df)}')
    print(df['agir_hasarli'].value_counts().rename({0: 'Hayır', 1: 'Evet'}))
    print()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(df)
    print(f'train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}')
    print(f'  train agir_hasarli=Evet: {(X_train["agir_hasarli"] == 1).sum()}')
    print(f'  val   agir_hasarli=Evet: {(X_val["agir_hasarli"] == 1).sum()}')
    print(f'  test  agir_hasarli=Evet: {(X_test["agir_hasarli"] == 1).sum()}')
    print()

    model = LGBMRegressor(**BASELINE_PARAMS)
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    report = segment_report(X_test, y_test, preds_test)

    display_df = report.copy()
    for col in ['gercek_ortalama_fiyat', 'tahmin_ortalamasi', 'bias', 'mae', 'rmse']:
        display_df[col] = display_df[col].map(lambda x: f'{x:,.0f}')
    display_df['r2'] = report['r2'].map(lambda x: f'{x:.4f}')
    print('=== Test seti (egitimde hic gorulmemis) - segment bazli performans ===')
    print(display_df.to_string(index=False))

    report_path = os.path.join(BASE_DIR, 'data', 'output', 'agir_hasarli_holdout_report.csv')
    report.to_csv(report_path, index=False)
    print(f'\nrapor kaydedildi: {report_path}')

    print()
    print('NOT: bu test SADECE "hic scrape edilmemis donem" kaynakli NaN etiketleri elemis '
          'olur. Kalan 0/1 etiketler icinde de src/utils/json-parser.js\'nin '
          '"alan bulunamadi -> 0" varsayilan mantigindan kaynaklanan, tespit edilemeyen bir '
          'yanlis-etiketleme riski hala var (bkz. modul docstring\'i).')


if __name__ == '__main__':
    main()
