"""Public dataset loader & normalizer (train split)."""
import os

import pandas as pd

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
