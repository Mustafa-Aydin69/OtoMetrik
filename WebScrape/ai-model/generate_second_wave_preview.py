"""Faz 31 ikinci dalga: 8 RECOMMENDED grup icin ONIZLEME sentetik veri.
train_dataset.csv'ye YAZMAZ, synthetic_pilot.csv'yi DEGISTIRMEZ, hicbir retrain
YAPMAZ. Ayri dosya: data/output/synthetic_second_wave_preview.csv.

Faz 30'un generate_synthetic_pilot.py mimarisi AYNEN korunur (_attempt_row):
sadece yil/kilometre/fiyat interpole edilir, motor_hacmi/motor_gucu/kategorik
alanlar TEK donor'dan kopyalanir, curve sapma siniri +-%12, gurultu +-%3
(parent sinirina kirpilir), her satir TAM 2 gercek parent'tan turer.

FARK (bu dalgaya ozgu, kullanicinin ekstra kontrol talebiyle): bazi gruplarda
GERCEK veri iki (veya daha fazla) ACIKCA FARKLI nesil/varyant iceriyor - bu
gruplarda pair secimi CLUSTER kisitlamasiyla yapilir, farkli nesiller
CAPRAZLANMAZ:

- Dodge Ram: 2004 satiri (kasa_turu=Hard top, 5883cc dizel - klasik pickup)
  ile 2021/2025 satirlari (kasa_turu=SUV, 2750cc - tamamen farkli govde/motor)
  AYNI arac DEGIL. 2004 satiri TEK ORNEK oldugu icin (kendi kumesinde es
  bulunamiyor) sentetik uretimden TAMAMEN haric tutulur - sadece 3 modern
  (SUV) satirdan, ARADAKI 3 essiz ciftten uretim yapilir (tekrar YOK,
  hedef 6 yerine 3 - gercek cesitlilik kadar).
- Lexus LS: 2015 satiri (4750.5cc - eski nesil LS600h/benzeri buyuk V8 hibrit)
  ile 2018-2022 satirlari (3250-3456cc - LS500h, 3.5L V6 turbo hibrit) FARKLI
  guc grubu/nesil. 2015 satiri haric tutulur, 6 sentetik SADECE 4 satirlik
  LS500h kumesinden (6 essiz cift mevcut, tekrarsiz).
- Bentley Flying Spur: fiyat/yil acikca IKI kumeye ayriliyor - 2013-2014
  (10.5-14.5M TL, 2. nesil) ve 2020-2022 (27.5-35M TL, 3. nesil, tamamen
  yeni platform). Iki kume ARASINDA cift KURULMAZ, her kumenin KENDI
  ICINDE cift secilir.
- Aston Martin Vantage / Mercedes Maybach S: motor_hacmi'ye gore iki alt-grup
  var (Vantage: erken 4.25L vs sonraki 4.7L; Maybach: S500 vs S560) - AYNI
  nameplate/govde nesli icinde bilinen bir motor guncellemesi (facelift),
  farkli ARAC DEGIL - yine de temkinlilik icin ayni kural uygulanir (cift
  SADECE ayni alt-grup icinde), cunku her iki alt-grup da yeterli (>=2)
  ornek icerdigi icin bu kisitlama hicbir cesitlilik kaybetmez.
- Rolls-Royce Wraith / Mercedes V Serisi: TEK nesil, kisitlama YOK.
- Audi RS: 'model' etiketi RS3/RS4/RS5/RS6/RS7 gibi TAMAMEN FARKLI araclari
  TEK grupta topluyor (bkz. asagidaki kanit) - synthetic_safe=False, bu
  dalgadan TAMAMEN CIKARILDI, sentetik satir URETILMEDI.

Calistirma (ai-model/ calisma dizini olarak): python generate_second_wave_preview.py
"""
import itertools
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocess import load_clean_train_dataset
from generate_synthetic_pilot import (
    DONOR_COLS, MAX_CURVE_DEVIATION_PCT, MAX_RETRIES_PER_ROW, NOISE_MAX_PCT, _attempt_row, _source_label,
)

SECOND_WAVE_SEED = 67890
GENERATED_AT = datetime.now(timezone.utc).isoformat()
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_second_wave_preview.csv')

# marka, model, [(cluster_adi, [ilan_id,...]), ...], hedef_sayi
PLAN = [
    ('Dodge', 'Ram', [
        ('modern_suv_2021_2025', ['arabam-36546117', 'arabam-32728964', 'arabam-39045432']),
    ], 6),  # hedef 6 ama sadece 3 essiz cift var -> 3 uretilecek, altta raporlanir
    ('Rolls-Royce', 'Wraith', [
        ('tek_nesil', ['kaggle-ab-2594', 'arabam-39045278', 'kaggle-ab-2598', 'kaggle-ab-2593',
                        'kaggle-ar-30610464', 'arabam-39045534']),
    ], 6),
    ('Lexus', 'LS', [
        ('ls500h_2018_2022', ['arabam-39592438', 'kaggle-ab-3', 'kaggle-ab-17', 'kaggle-ab-18']),
    ], 6),  # 2015 (4750.5cc, eski nesil) HARIC
    ('Aston Martin', 'Vantage', [
        ('early_4.25L', ['kaggle-ab-434', 'kaggle-ab-442']),
        ('later_4.7L', ['kaggle-ab-438', 'kaggle-ab-443', 'arabam-26164209', 'kaggle-ab-441']),
    ], 6),
    ('Bentley', 'Flying Spur', [
        ('gen2_2013_2014', ['kaggle-ab-272', 'kaggle-ab-271', 'arabam-42433249', 'arabam-39563948']),
        ('gen3_2020_2022', ['arabam-39320430', 'kaggle-ar-30608246', 'arabam-40722555']),
    ], 6),
    ('Mercedes - Benz', 'Maybach S', [
        ('s500', ['arabam-41549788', 'arabam-26227468']),
        ('s560', ['arabam-42317572', 'kaggle-ab-22077', 'kaggle-ab-19954', 'kaggle-ab-19950']),
    ], 6),
    ('Mercedes - Benz', 'V Serisi', [
        ('tek_havuz', None),  # None = grubun TUM gercek satirlari (homojen kabul edildi)
    ], 5),
]
# Audi RS: PLAN'da YOK - heterojen model etiketi nedeniyle bu dalgadan cikarildi (bkz. main()).

MAX_PER_GROUP = {'Dodge|Ram': 6, 'Rolls-Royce|Wraith': 6, 'Lexus|LS': 6, 'Aston Martin|Vantage': 6,
                  'Bentley|Flying Spur': 6, 'Mercedes - Benz|Maybach S': 6, 'Mercedes - Benz|V Serisi': 5}


def generate_clustered(marka, model, clusters, n_target, rng, hp_lookup, clean_df):
    full_group = clean_df[(clean_df['marka'] == marka) & (clean_df['model'] == model)].set_index('ilan_id')
    all_pairs = []  # (cluster_adi, i_id, j_id)
    for cluster_adi, ids in clusters:
        if ids is None:
            ids = list(full_group.index)
        ids = [i for i in ids if i in full_group.index]
        for i_id, j_id in itertools.combinations(ids, 2):
            all_pairs.append((cluster_adi, i_id, j_id))

    if not all_pairs:
        return [], 0

    # Faz 31 duzeltmesi: sabit N cift secip SADECE t'yi retry etmek yerine,
    # TUM kume-ici ciftleri karistirip sirayla dener - bir ciftin HICBIR t
    # degeri guvenli araliga girmiyorsa (curve guard), digger ciftlere GECER.
    # V Serisi'nde ilk versiyonda 5 hedeften sadece 1'i uretilebilmisti cunku
    # onceden secilmis 5 cift sabitti, alternatif cift denenmiyordu.
    order = list(rng.permutation(len(all_pairs)))

    out = []
    k = 0
    for pidx in order:
        if len(out) >= n_target:
            break
        cluster_adi, i_id, j_id = all_pairs[pidx]
        ri, rj = full_group.loc[i_id], full_group.loc[j_id]
        ri = ri.copy(); ri['ilan_id'] = i_id
        rj = rj.copy(); rj['ilan_id'] = j_id

        accepted = None
        for _ in range(MAX_RETRIES_PER_ROW):
            row, in_range, curve_ok = _attempt_row(marka, model, ri, rj, rng, hp_lookup)
            if in_range and curve_ok:
                accepted = row
                break
        if accepted is None:
            print(f'  UYARI: {marka} {model} [{cluster_adi}] cift ({i_id}+{j_id}) hicbir t degerinde guvenli araliga girmedi, ATLANDI')
            continue

        k += 1
        accepted['ilan_id'] = f'synthetic2-{marka}-{model}-{cluster_adi}-{k}'.replace(' ', '_')
        accepted['marka'] = marka
        accepted['model'] = model
        accepted['cluster'] = cluster_adi
        accepted['is_synthetic'] = 1
        accepted['source'] = 'synthetic_second_wave_preview'
        accepted['synthetic_method'] = 'log_linear_2parent_interp(yil,km,fiyat)+single_donor_categoricals+bounded_noise+curve_guard+generation_cluster_restricted'
        accepted['synthetic_seed'] = SECOND_WAVE_SEED
        accepted['generated_at'] = GENERATED_AT
        accepted['synthetic_safe_check'] = True
        accepted['generation_cluster_restricted'] = len(clusters) > 1 or clusters[0][1] is not None
        out.append(accepted)
    return out, len(all_pairs)


def main():
    print('Production preprocessing ile TEMIZ (gercek) veri okunuyor (SADECE OKUMA)...')
    clean = load_clean_train_dataset()

    print('Production hierarchical_price artefakti okunuyor (SADECE OKUMA)...')
    import joblib
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'lightgbm_final.joblib'))
    hp_lookup = artifact['hierarchical_price']

    rng = np.random.default_rng(SECOND_WAVE_SEED)
    all_rows = []
    for marka, model, clusters, n_target in PLAN:
        print(f'\n{marka} {model}: hedef {n_target}, kume sayisi={len(clusters)}')
        generated, n_pairs_available = generate_clustered(marka, model, clusters, n_target, rng, hp_lookup, clean)
        print(f'  essiz (kume-ici) cift sayisi: {n_pairs_available}, uretilen: {len(generated)}')
        if len(generated) < n_target:
            print(f'  NOT: hedef {n_target} idi, sadece {len(generated)} uretildi (essiz cift sinirlamasi/guvenlik reddi)')
        all_rows.extend(generated)

    print('\nAudi RS: HETEROJEN model etiketi nedeniyle bu dalgadan CIKARILDI (0 satir), bkz. rapor.')

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\nYAZILDI (train_dataset.csv/synthetic_pilot.csv DEGISMEDI, retrain YAPILMADI): {os.path.abspath(OUT_PATH)}')
    print(f'Toplam sentetik satir: {len(df)}')
    print(df.groupby(['marka', 'model']).size().to_string())


if __name__ == '__main__':
    main()
