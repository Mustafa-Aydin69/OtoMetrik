"""Faz 30 pilot: Ferrari 458 / Lamborghini Huracan / Rolls-Royce Ghost icin
GUVENLI interpolasyon noktalarinin ONIZLEMESINI ureten script.

BU SCRIPT train_dataset.csv'ye HICBIR SATIR EKLEMEZ, hicbir retrain tetiklemez -
sadece reports/pilot_synthetic_preview.csv'ye "eger uretilirse boyle gorunur"
onizlemesi yazar. Kullanicinin onayi olmadan hicbir kalici veri degisikligi
YAPILMAZ.

YONTEM (kullanicinin guvenlik kurallarina birebir uyar):
- Her sentetik satir, AYNI grubun GERCEK iki satiri (i, j) arasinda kurulur.
  yil/kilometre DUZ (linear), fiyat LOG-LINEAR (Theil-Sen/hierarchical_price.py
  ile AYNI varsayim - fiyat yasla CARPIMSAL degisir) interpolasyon ile
  t~Uniform(0,1) oraninda hesaplanir - t HER ZAMAN [0,1] icinde oldugundan
  sonuc otomatik olarak [i,j] ARALIGINDA kalir, gruplarin gozlemlenen
  [min,max] araligi DISINA CIKAMAZ (extrapolation yok).
- paket/kasa_turu/renk/motor_hacmi/motor_gucu/yakit_turu/vites/degisen_sayisi/
  boyali_sayisi/agir_hasarli gibi kategorik/durum alanlari INTERPOLE EDILMEZ -
  t<0.5 ise i'nin, t>=0.5 ise j'nin TAM kategorik seti KOPYALANIR. Boylece
  gercekte hic birlikte gorulmemis bir kombinasyon ICAT EDILMEZ.
- source='synthetic', is_synthetic=1, source_parent_ids='{id_i}+{id_j}'.
- SABIT seed (PILOT_SEED) - tekrar calistirildiginda AYNI noktalar uretilir.
- Cift (i,j) sayisi grubun gercek satir sayisindan azsa (orn. Huracan n=2 ->
  tek cift), AYNI cift farkli t degerleriyle TEKRAR kullanilir - bu da
  interpolasyon araligini DEGISTIRMEZ, sadece o aralik icinde baska bir nokta
  secer.

Calistirma (ai-model/ calisma dizini olarak): python generate_pilot_synthetic_preview.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import load_clean_train_dataset

PILOT_SEED = 42
PILOT_PLAN = [
    ('Ferrari', '458', 8),
    ('Lamborghini', 'Huracan', 4),
    ('Rolls-Royce', 'Ghost', 6),
]
CATEGORICAL_DONOR_COLS = [
    'paket', 'kasa_turu', 'renk', 'motor_hacmi', 'motor_gucu',
    'yakit_turu', 'vites', 'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli',
]
OUT_PATH = os.path.join(os.path.dirname(__file__), 'reports', 'pilot_synthetic_preview.csv')


def generate_group(marka, model, real_rows, n_target, rng):
    real_rows = real_rows.reset_index(drop=True)
    n_real = len(real_rows)
    pairs = list(itertools.combinations(range(n_real), 2))
    if not pairs:
        return []

    # cift secimi: mevcut ciftler tukenmeden AYNI cift tekrar kullanilmaz (n_target
    # <= C(n_real,2) ise); tukenirse (kucuk gruplar) rastgele TEKRAR secilir.
    if n_target <= len(pairs):
        chosen_pair_idx = rng.choice(len(pairs), size=n_target, replace=False)
    else:
        chosen_pair_idx = rng.choice(len(pairs), size=n_target, replace=True)

    out = []
    for k, pidx in enumerate(chosen_pair_idx):
        i, j = pairs[pidx]
        ri, rj = real_rows.iloc[i], real_rows.iloc[j]
        t = float(rng.uniform(0.15, 0.85))  # ucundan degil, ic bolgeden ornek - gercek uc noktalari TEKRARLAMAMAK icin

        yil = round(ri['yil'] + t * (rj['yil'] - ri['yil']))
        kilometre = round((ri['kilometre'] + t * (rj['kilometre'] - ri['kilometre'])) / 500) * 500
        fiyat = float(np.exp(np.log(ri['fiyat']) + t * (np.log(rj['fiyat']) - np.log(ri['fiyat']))))
        fiyat = round(fiyat / 10_000) * 10_000

        donor = ri if t < 0.5 else rj
        donor_label = 'i' if t < 0.5 else 'j'

        row = {
            'ilan_id': f'synthetic-{marka}-{model}-{k+1}'.replace(' ', '_'),
            'marka': marka, 'model': model,
            'yil': yil, 'kilometre': kilometre, 'fiyat': fiyat,
            't': round(t, 3),
            'source': 'synthetic', 'is_synthetic': 1,
            'source_parent_ids': f"{ri['ilan_id']}+{rj['ilan_id']}",
            'categorical_donor_id': donor['ilan_id'],
            'categorical_donor_side': donor_label,
        }
        for c in CATEGORICAL_DONOR_COLS:
            row[c] = donor[c]

        # guvenlik dogrulamasi: uretilen deger grubun GERCEK gozlem araligi
        # DISINA cikmis mi (extrapolation kontrolu) - asla True olmamali,
        # olursa satir REDDEDILIR (uretilmez).
        in_range = (
            real_rows['yil'].min() <= yil <= real_rows['yil'].max() and
            real_rows['kilometre'].min() <= kilometre <= real_rows['kilometre'].max() and
            real_rows['fiyat'].min() <= fiyat <= real_rows['fiyat'].max()
        )
        row['range_check_passed'] = in_range
        if in_range:
            out.append(row)
        else:
            print(f'  UYARI: {marka} {model} #{k+1} araligi disina cikti, ATLANDI (yil={yil}, km={kilometre}, fiyat={fiyat})')
    return out


def main():
    print('Production preprocessing ile TEMIZ veri okunuyor (SADECE OKUMA)...')
    clean = load_clean_train_dataset()

    rng = np.random.default_rng(PILOT_SEED)
    all_rows = []
    for marka, model, n_target in PILOT_PLAN:
        real_rows = clean[(clean['marka'] == marka) & (clean['model'] == model)]
        print(f'\n{marka} {model}: {len(real_rows)} gercek satir, hedef {n_target} sentetik')
        generated = generate_group(marka, model, real_rows, n_target, rng)
        print(f'  uretilen (araligi gecen): {len(generated)}')
        all_rows.extend(generated)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\nONIZLEME yazildi (train_dataset.csv DEGISMEDI, retrain YAPILMADI): {OUT_PATH}')
    print(f'Toplam onerilen sentetik satir: {len(df)}')
    print('\nOzet:')
    print(df.groupby(['marka', 'model']).size().to_string())


if __name__ == '__main__':
    main()
