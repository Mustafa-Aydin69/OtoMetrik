"""Public dataset loader & normalizer (train split)."""
import os

import pandas as pd

from utils.text_cleaner import (
    parse_boya_degisen,
    parse_kilometre,
    parse_number,
    parse_price_tl,
    parse_turkish_date,
)

SCHEMA_FIELDS = [
    'ilan_id', 'arac_turu', 'marka', 'model', 'paket', 'kasa_turu', 'renk',
    'motor_hacmi', 'motor_gucu', 'yil', 'kilometre', 'yakit_turu', 'vites',
    'degisen_sayisi', 'boyali_sayisi', 'fiyat', 'scraped_at',
]

KAGGLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'kaggle')


# araba_bilgileri.csv zaten snake_case ve buyuk olcude sayisal; dogrudan alan eslemesi yeterli.
# arac_turu ve scraped_at bu kaynakta yok; scraped_at icin sahte bir tarih uydurmak yerine
# bilinmiyor (None) birakiyoruz (arac_turu'nde de ayni "bilinmiyorsa None" kurali uygulanmisti).
def load_araba_bilgileri():
    df = pd.read_csv(os.path.join(KAGGLE_DIR, 'araba_bilgileri.csv'))
    return pd.DataFrame({
        'ilan_id': 'kaggle-ab-' + df['id'].astype(str),
        'arac_turu': None,
        'marka': df['marka'],
        'model': df['seri'],
        'paket': df['model'],
        'kasa_turu': df['kasa_tipi'],
        'renk': df['renk'],
        'motor_hacmi': df['motor_hacmi'],
        'motor_gucu': df['motor_gucu'],
        'yil': df['yil'],
        'kilometre': df['kilometre'],
        'yakit_turu': df['yakit_tipi'],
        'vites': df['vites_tipi'],
        'degisen_sayisi': df['degisen_sayisi'],
        'boyali_sayisi': df['boyali_sayisi'],
        'fiyat': df['fiyat'],
        'scraped_at': None,
    })[SCHEMA_FIELDS]


# arabalar.csv arabam.com'dan kazinmis; Ilan No'daki "Kopyalandi" onekini (bizim kendi
# scraper'imizda bulup duzelttigimiz ayni kopyalama-tooltip kirliligi) temizliyoruz.
def load_arabalar():
    df = pd.read_csv(os.path.join(KAGGLE_DIR, 'arabalar.csv'), encoding='utf-8-sig')

    ilan_id = df['İlan No'].astype(str).str.replace(r'\D', '', regex=True)
    boya_degisen = df['Boya-değişen'].apply(
        lambda v: parse_boya_degisen(v) if isinstance(v, str) else (None, None)
    )
    degisen_sayisi, boyali_sayisi = zip(*boya_degisen)

    return pd.DataFrame({
        'ilan_id': 'kaggle-ar-' + ilan_id,
        'arac_turu': None,
        'marka': df['Marka'],
        'model': df['Seri'],
        'paket': df['Model'],
        'kasa_turu': df['Kasa Tipi'],
        'renk': df['Renk'],
        'motor_hacmi': df['Motor Hacmi'].apply(parse_number),
        'motor_gucu': df['Motor Gücü'].apply(parse_number),
        'yil': df['Yıl'],
        'kilometre': df['Kilometre'].apply(parse_kilometre),
        'yakit_turu': df['Yakıt Tipi'],
        'vites': df['Vites Tipi'],
        'degisen_sayisi': degisen_sayisi,
        'boyali_sayisi': boyali_sayisi,
        'fiyat': df['Fiyat'].apply(parse_price_tl),
        'scraped_at': df['İlan Tarihi'].apply(parse_turkish_date),
    })[SCHEMA_FIELDS]


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'train_dataset.csv')


def build_train_dataset():
    return pd.concat([load_araba_bilgileri(), load_arabalar()], ignore_index=True)


def main():
    df = build_train_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f'{len(df)} kayit yazildi: {OUTPUT_PATH}')
    print('Kaynak dagilimi:')
    print(df['ilan_id'].str.extract(r'^kaggle-(\w+)-')[0].value_counts().to_string())
    print('Eksik deger orani (%):')
    print((df.isna().mean() * 100).round(1).to_string())


if __name__ == '__main__':
    main()
