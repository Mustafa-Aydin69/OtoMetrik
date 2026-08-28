"""Faz 9 Madde 1: Faz 8 EDA kararlarini uygulayan yeniden calistirilabilir on-isleme pipeline'i.

Uygulanan kararlar (bkz. eda_and_preprocessing.ipynb Madde 5):
- marka-ici ust %1 fiyat + 1.000.000 km ustu kayitlar cikarilir (veri kirliligi;
  Faz 29 - bkz. asagidaki not, GLOBAL degil marka bazinda).
- arac_turu / scraped_at cikarilir (train_dataset'te sirasiyla %100 / %95 eksik, sinyal tasimaz).
- degisen_sayisi / boyali_sayisi NaN'i "bilinmiyor" ayri flag kolonuyla isaretlenir, sonra 0'a doldurulur
  (dogrudan 0'a doldurmak "hasarsiz" gibi yanlis sinyal verirdi).
- agir_hasarli NaN'i (kullanici karari, gecici) dogrudan 0'a (hasarsiz) doldurulur - daha genis
  kazima sonrasi bu basitlestirme fine-tuning ile duzeltilecek.
- yas = PRICE_REFERENCE_DATE.year - yil, km_yil = kilometre / yas turetilir.
- Geri kalan kucuk oranli eksik degerler (marka/model/yakit_turu/vites/motor_hacmi/motor_gucu/yil)
  baseline asamasinda satir bazinda cikarilir.

Faz 29: fiyat aykiri-deger kirpmasi GLOBAL tek esikten (tum 308k satira gore %99) MARKA-ICI
esige gecti. Global esik ucuz/orta segment agirlikli pazara gore hesaplaniyordu (~4.85M TL) -
Ferrari (30 ham satir, en ucuzu bile bu esigin ustunde) ve Lamborghini (7 satir) gibi premium
markalarin TAMAMINI, Bentley/Rolls-Royce'u da neredeyse tamamini silip egitim verisinden
komple dusuruyordu (bkz. hierarchical_price.py Faz 29 notu - bu markalar icin model artik
dogru fiyat tahmini uretemiyordu). Marka-ici %99'luk dilim o markanin KENDI fiyat dagilimina
gore aykiri degerleri (veri girisi hatasi supheli) elerken, marka'nin gercek/gecerli fiyat
araligini (ornegin bir Ferrari'nin milyonlarca TL olmasi) "aykiri" saymaz.

Faz 29: PRICE_REFERENCE_DATE tek zaman kaynagi. Onceden CURRENT_YEAR sabit 2026'ydi ve
hierarchical_price.py kendi date.today() cagrisini ayri yapiyordu - iki farkli "simdi"
kavrami ayni artefaktta yasayabiliyordu. Artik ikisi de bu moduldeki TEK date.today()
cagrisindan turetiliyor (bkz. hierarchical_price.py'nin PRICE_REFERENCE_DATE importu).

Faz 30: final dropna() ARTIK TUM kolonlara degil, sadece REQUIRED_COLS'a uygulanir.
Kok neden (bkz. analyze_dropna_loss.py raporu): eski blanket dropna() 17.382 satiri
(%5.70) siliyordu, bunun 1467'si SADECE motor_hacmi NaN oldugu icin, 896'si SADECE
paket NaN oldugu icin - marka+model gecerli, fiyat/yil/km gecerli olsa BILE satir
TAMAMEN atiliyordu. Bu, 256 marka+model grubunu (Ferrari 458 haric tutulan bazi nadir
modeller dahil) egitimden TAMAMEN dusuruyordu (raw_real_count>0, train_real_count=0).
3 kademeli ayrim:
  - REQUIRED_COLS (fiyat/yil/kilometre/marka/model): eksikse satir GERCEKTEN dusurulur
    - target yoksa egitilemez, yil/km olmadan yas/km_yil/hierarchical_price kurulamaz,
      marka/model olmadan grup/kategori tanimsizdir.
  - CATEGORICAL_FILLNA_COLS (paket/kasa_turu/renk/yakit_turu/vites): eksikse satir
    DUSURULMEZ, sabit 'Belirtilmemiş' kategorisine cevrilir - degisen_sayisi_bilinmiyor/
    boyali_sayisi_bilinmiyor ile AYNI "bilinmiyor'u ayri bir deger olarak isaretle" deseni.
    serve.py'de 'paket' (trim) ayni sabit degere duser (bkz. serve.py Faz 30 notu) -
    train/inference AYNI "bilinmiyor" temsilini kullanir.
  - motor_hacmi/motor_gucu: NE dusurulur NE doldurulur - NaN oldugu gibi birakilir,
    LightGBM'in native missing-value isleyisine (bkz. train.py'nin to_category()
    yorumu - "gorulmemis kategoriler native missing olur", ayni tolerans burada
    numeric icin de uygulanir) devredilir. 0/medyan ile doldurmak YANLIS sinyal
    verirdi (orn. motor_hacmi=0 elektrikli araclarda ZATEN anlamli bir sentinel,
    bkz. generate_vehicle_options.py'nin hacmi_bucket=0.0 kullanimi - motor_hacmi
    NaN'ini 0 ile doldurmak bu iki farkli anlami CARPISTIRIRDI).
"""
import os
from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split

PRICE_REFERENCE_DATE = date.today()
CURRENT_YEAR = PRICE_REFERENCE_DATE.year
DROP_COLS = ['arac_turu', 'scraped_at']
UNKNOWN_FLAG_COLS = ['degisen_sayisi', 'boyali_sayisi']
ONEHOT_COLS = ['yakit_turu', 'vites', 'kasa_turu']
FREQ_COLS = ['marka', 'model', 'paket', 'renk']

# Faz 30 (bkz. modul docstring): final dropna()'nin ARTIK sadece bu alt kumeye
# uygulandigi zorunlu alanlar - eksikse satir gercekten egitilemez.
REQUIRED_COLS = ['fiyat', 'yil', 'kilometre', 'marka', 'model']
# Eksikse satir DUSURULMEZ - sabit 'Belirtilmemiş' kategorisine cevrilir.
CATEGORICAL_FILLNA_COLS = ['paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites']
UNKNOWN_CATEGORY_VALUE = 'Belirtilmemiş'

CATALOG_CATEGORY_COLS = ['marka', 'model', 'paket', 'kasa_turu', 'yakit_turu', 'vites']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv')


def load_catalog_dataset():
    """UI arac secim kataloğu (generate_vehicle_options.py -> vehicle-options.generated.ts)
    icin ayri veri kaynagi. load_clean_train_dataset()'ten kasitli olarak FARKLI: fiyat/km
    q99 kirpmasi UYGULANMAZ. O kirpma TUM pazara (308k satir, ucuz/orta segment agirlikli)
    gore hesaplanan tek bir global esikti - Ferrari (30 ham satir, en ucuzu bile esigin
    ustunde) ve Lamborghini (7 satir) gibi premium markalarin TAMAMINI silip dropdown'dan
    tamamen dusuruyordu, Bentley/Rolls-Royce'u da tek modele indiriyordu. Model egitim
    pipeline'ina (train.py, hierarchical_price.py, load_clean_train_dataset()) HIC baglanmaz
    - iki fonksiyon TRAIN_PATH'i ayri ayri okur, birbirinin sonucunu etkilemez.
    """
    df = pd.read_csv(TRAIN_PATH, low_memory=False)

    df = df.drop_duplicates(subset=['ilan_id'], keep='first')
    df = df.dropna(subset=['marka', 'model'])

    for col in CATALOG_CATEGORY_COLS:
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()

    return df.reset_index(drop=True)


def load_clean_train_dataset():
    df = pd.read_csv(TRAIN_PATH, low_memory=False)

    brand_q99 = df.groupby('marka')['fiyat'].transform(lambda s: s.quantile(0.99))
    df = df[df['fiyat'] <= brand_q99]
    df = df[df['kilometre'] <= 1_000_000]

    df = df.drop(columns=DROP_COLS)

    for col in UNKNOWN_FLAG_COLS:
        df[f'{col}_bilinmiyor'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    # agir_hasarli %95.6 eksik (araba_bilgileri.csv'de bu alan hic yok, eski canli kazima kayitlari
    # bu alani henuz aramiyordu). Kullanici karari: bu turda bilinmeyeni "hasarsiz" (0) sayip
    # egit, daha genis kazima (agir_hasarli isaretli) sonrasi fine-tuning ile duzeltilecek.
    df['agir_hasarli'] = df['agir_hasarli'].fillna(0)

    # Faz 30: kategorik yardimci alanlar eksikse satir dusurulmez - sabit
    # UNKNOWN_CATEGORY_VALUE'ya cevrilir (bkz. modul docstring'i).
    for col in CATEGORICAL_FILLNA_COLS:
        df[col] = df[col].fillna(UNKNOWN_CATEGORY_VALUE)

    df['yas'] = CURRENT_YEAR - df['yil']
    df['km_yil'] = df['kilometre'] / df['yas'].replace(0, 1)

    # Faz 30: dropna ARTIK sadece REQUIRED_COLS'a bakar - motor_hacmi/motor_gucu
    # KASITLI olarak burada YOK, NaN kalirlar (bkz. modul docstring'i).
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    print(f'Zorunlu alanlar (REQUIRED_COLS) icin {before - len(df)} satir cikarildi '
          f'({100 * (before - len(df)) / before:.2f}%)')

    return df


def split_features_target(df, test_size=0.2, random_state=42):
    y = df['fiyat']
    X = df.drop(columns=['fiyat', 'ilan_id'])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Dusuk kardinaliteli alanlar one-hot, yuksek kardiniteli alanlar (marka/model/paket/renk)
# frekans encoding ile sayisallastirilir. Sizinti olmasin diye harita X_train'e gore fit edilir,
# X_test'e sadece transform uygulanir.
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


def main():
    df = load_clean_train_dataset()
    print(f'{len(df)} kayit (temizlenmis + turetilmis ozellikler dahil)')
    print(df.dtypes.astype(str).to_string())

    X_train, X_test, y_train, y_test = split_features_target(df)
    print(f'train: {len(X_train)}, test: {len(X_test)}')


if __name__ == '__main__':
    main()
