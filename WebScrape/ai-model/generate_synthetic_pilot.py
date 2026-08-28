"""Faz 30 pilot v3 (kullanici duzeltmeleri): Ferrari 458 / Lamborghini Huracan /
Rolls-Royce Ghost icin KONTROLLU sentetik veri uretir. train_dataset.csv'ye
YAZMAZ - AYRI dosya: data/output/synthetic_pilot.csv. Bu script'i calistirmak
train_dataset.csv'yi DEGISTIRMEZ, hicbir model retrain etmez.

v3 DUZELTMELERI (v2'den farklar):
- motor_hacmi/motor_gucu ARTIK INTERPOLE EDILMEZ - teknik spesifikasyon oldugu
  icin (kullanici karari) DONOR_COLS'a tasindi, TEK parent'tan aynen kopyalanir.
- SADECE yil/kilometre/fiyat interpole edilir.
- hierarchical_price egri sapmasi icin GUVENLI SINIR: |curve_deviation_pct| <=
  MAX_CURVE_DEVIATION_PCT (%12). Asan satir REDDEDILIR, AYNI rng akisindan
  (deterministik) yeni bir (pair,t) denenir - MAX_RETRIES'e kadar.
- fiyat ayristirmasi: interpolation_base_price (gurultusuz log-lineer parent
  interpolasyonu) / random_noise_pct (CEKILEN gurultu, kirpmadan ONCE) /
  final_price (kirpma SONRASI, = 'fiyat') / curve_deviation_pct (final_price'in
  hierarchical_price egrisinden sapmasi) AYRI sutunlar - hangi asamada ne kadar
  degistigi izlenebilir.
- parent_source_1/parent_source_2 ('arabam'/'kaggle') + contains_current_scrape.

KURALLAR (degismeyenler):
- Her sentetik satir TAM OLARAK 2 gercek parent'tan (i,j) turer.
- yil/kilometre: t~Uniform(0.15,0.85) ile parent'lar ARASINDA lineer
  interpolasyon - sonuc [parent_i,parent_j] araligina KIRPILIR (extrapolation yok).
- paket/kasa_turu/renk/yakit_turu/vites/motor_hacmi/motor_gucu/degisen_sayisi/
  boyali_sayisi/agir_hasarli: INTERPOLE EDILMEZ - t<0.5 ise i'nin, t>=0.5 ise
  j'nin TAM kombinasyonu KOPYALANIR (parent A'nin paketiyle parent B'nin motoru
  gibi bir CAPRAZLAMA YAPILMAZ - hepsi AYNI donor'dan gelir).
- SABIT seed (PILOT_SEED) - tekrar calistirilinca AYNI satirlar uretilir.
- Metadata: is_synthetic=1, source='synthetic_pilot', source_parent_ids,
  synthetic_method, synthetic_seed, generated_at.

Calistirma (ai-model/ calisma dizini olarak): python generate_synthetic_pilot.py
"""
import itertools
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import CURRENT_YEAR, load_clean_train_dataset
import hierarchical_price as hp

PILOT_SEED = 12345
NOISE_MAX_PCT = 0.03
MAX_CURVE_DEVIATION_PCT = 12.0
MAX_RETRIES_PER_ROW = 40
PILOT_PLAN = [
    ('Ferrari', '458', 8),
    ('Lamborghini', 'Huracan', 4),
    ('Rolls-Royce', 'Ghost', 6),
]
# v3: motor_hacmi/motor_gucu ARTIK burada - interpole EDILMEZ, donor'dan kopyalanir.
DONOR_COLS = ['paket', 'kasa_turu', 'renk', 'yakit_turu', 'vites', 'motor_hacmi', 'motor_gucu',
              'degisen_sayisi', 'boyali_sayisi', 'agir_hasarli']
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_pilot.csv')
GENERATED_AT = datetime.now(timezone.utc).isoformat()


def _source_label(ilan_id):
    return 'arabam' if str(ilan_id).startswith('arabam-') else ('kaggle' if str(ilan_id).startswith('kaggle-') else 'other')


def _attempt_row(marka, model, ri, rj, rng, hp_lookup):
    t = float(rng.uniform(0.15, 0.85))

    lo_yil, hi_yil = sorted([ri['yil'], rj['yil']])
    lo_km, hi_km = sorted([ri['kilometre'], rj['kilometre']])
    lo_p, hi_p = sorted([ri['fiyat'], rj['fiyat']])

    yil = min(hi_yil, max(lo_yil, round(ri['yil'] + t * (rj['yil'] - ri['yil']))))
    kilometre = min(hi_km, max(lo_km, round((ri['kilometre'] + t * (rj['kilometre'] - ri['kilometre'])) / 500) * 500))

    interpolation_base_price = float(np.exp(np.log(ri['fiyat']) + t * (np.log(rj['fiyat']) - np.log(ri['fiyat']))))
    noise = float(rng.uniform(-NOISE_MAX_PCT, NOISE_MAX_PCT))
    noised_price = interpolation_base_price * (1 + noise)
    final_price = min(hi_p, max(lo_p, round(noised_price / 10_000) * 10_000))
    noise_clipped = round(noised_price / 10_000) * 10_000 != final_price

    yas = max(CURRENT_YEAR - yil, 0)
    curve_val, curve_src, curve_n = hp.lookup_price(marka, model, yas, hp_lookup)
    curve_deviation_pct = round(100 * (final_price - curve_val) / curve_val, 2) if curve_val else None

    donor = ri if t < 0.5 else rj
    donor_side = 'i' if t < 0.5 else 'j'

    in_range = lo_yil <= yil <= hi_yil and lo_km <= kilometre <= hi_km and lo_p <= final_price <= hi_p
    curve_ok = curve_deviation_pct is not None and abs(curve_deviation_pct) <= MAX_CURVE_DEVIATION_PCT

    row = {
        't': round(t, 3), 'yil': yil, 'kilometre': kilometre,
        'interpolation_base_price': round(interpolation_base_price, 0),
        'random_noise_pct': round(noise * 100, 2),
        'final_price': final_price, 'fiyat': final_price,
        'noise_effective_pct': round(100 * (final_price - interpolation_base_price) / interpolation_base_price, 2),
        'noise_clipped_by_parent_bound': bool(noise_clipped),
        'curve_deviation_pct': curve_deviation_pct,
        'hp_curve_value': round(curve_val, 0), 'hp_curve_source': curve_src,
        'source_parent_ids': f"{ri['ilan_id']}+{rj['ilan_id']}",
        'parent_source_1': _source_label(ri['ilan_id']), 'parent_source_2': _source_label(rj['ilan_id']),
        'contains_current_scrape': _source_label(ri['ilan_id']) == 'arabam' or _source_label(rj['ilan_id']) == 'arabam',
        'categorical_donor_id': donor['ilan_id'], 'categorical_donor_side': donor_side,
    }
    for c in DONOR_COLS:
        row[c] = donor[c]
    return row, in_range, curve_ok


def generate_group(marka, model, real_rows, n_target, rng, hp_lookup):
    real_rows = real_rows.reset_index(drop=True)
    n_real = len(real_rows)
    pairs = list(itertools.combinations(range(n_real), 2))
    if not pairs:
        return []

    if n_target <= len(pairs):
        chosen_pair_idx = list(rng.choice(len(pairs), size=n_target, replace=False))
    else:
        chosen_pair_idx = list(rng.choice(len(pairs), size=n_target, replace=True))

    out = []
    for k, pidx in enumerate(chosen_pair_idx):
        i, j = pairs[pidx]
        ri, rj = real_rows.iloc[i], real_rows.iloc[j]

        accepted = None
        for attempt in range(MAX_RETRIES_PER_ROW):
            row, in_range, curve_ok = _attempt_row(marka, model, ri, rj, rng, hp_lookup)
            if in_range and curve_ok:
                accepted = row
                break
        if accepted is None:
            print(f'  UYARI: {marka} {model} #{k+1} - {MAX_RETRIES_PER_ROW} denemede guvenli araliga girmedi, ATLANDI '
                  f'(son deneme curve_deviation={row["curve_deviation_pct"]}%, in_range={in_range})')
            continue

        accepted['ilan_id'] = f'synthetic-{marka}-{model}-{k+1}'.replace(' ', '_')
        accepted['marka'] = marka
        accepted['model'] = model
        accepted['is_synthetic'] = 1
        accepted['source'] = 'synthetic_pilot'
        accepted['synthetic_method'] = 'log_linear_2parent_interp(yil,km,fiyat)+single_donor_categoricals+bounded_noise+curve_guard'
        accepted['synthetic_seed'] = PILOT_SEED
        accepted['generated_at'] = GENERATED_AT
        accepted['range_check_passed'] = True
        out.append(accepted)
    return out


def main():
    print('Production preprocessing ile TEMIZ (gercek) veri okunuyor (SADECE OKUMA)...')
    clean = load_clean_train_dataset()

    print('Production hierarchical_price artefakti okunuyor (SADECE OKUMA, tutarlilik+guvenlik siniri icin)...')
    import joblib
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'lightgbm_final.joblib'))
    hp_lookup = artifact['hierarchical_price']

    rng = np.random.default_rng(PILOT_SEED)
    all_rows = []
    for marka, model, n_target in PILOT_PLAN:
        real_rows = clean[(clean['marka'] == marka) & (clean['model'] == model)]
        print(f'\n{marka} {model}: {len(real_rows)} gercek satir (parent havuzu), hedef {n_target} sentetik')
        generated = generate_group(marka, model, real_rows, n_target, rng, hp_lookup)
        print(f'  uretilen (araligi VE curve guvenlik sinirini gecen): {len(generated)}')
        all_rows.extend(generated)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\nYAZILDI (train_dataset.csv DEGISMEDI, retrain YAPILMADI): {os.path.abspath(OUT_PATH)}')
    print(f'Toplam sentetik satir: {len(df)}')
    print(df.groupby(['marka', 'model']).size().to_string())
    print(f'\nnoise_clipped_by_parent_bound=True olan satir sayisi: {int(df["noise_clipped_by_parent_bound"].sum())}/{len(df)}')
    print(f'contains_current_scrape=True olan satir sayisi: {int(df["contains_current_scrape"].sum())}/{len(df)}')


if __name__ == '__main__':
    main()
