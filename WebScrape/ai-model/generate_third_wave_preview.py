"""Faz 34 pilot v3: Recommended Third-Wave Pilot (6 model) icin ONIZLEME
sentetik veri. train_dataset.csv/synthetic_pilot.csv/synthetic_second_wave_
preview.csv'ye DOKUNMAZ, hicbir retrain YAPMAZ. Ayri dosya:
data/output/synthetic_third_wave_preview.csv.

Mimari Faz30/31 ile AYNI (bkz. generate_synthetic_pilot.py._attempt_row):
sadece yil/km/fiyat interpole, motor_hacmi/motor_gucu/kategorik alanlar TEK
donor'dan aynen kopya, curve guard +-%12 (retry/reject), gurultu +-%3 (parent
sinirina kirpilir), her satir TAM 2 gercek parent'tan turer.

KUME TANIMLARI (Faz34 gercek-veri incelemesinden, HARDCODE isim DEGIL - gercek
motor_hacmi/motor_gucu/yil degerlerine dayanir):

- Audi TTS: anomali satiri (arabam-40704969, 2012, 2250.5cc/463hp - standart
  TTS spesifikasyonlarina uymuyor) HAVUZ DISI. Kalanlar iki kumeye ayrilir:
  mk2 (2008-2012, 1900-1984cc/263-272hp, 7 satir) ve mk3 (2016, 1900.5cc/313hp,
  TEK satir - es bulunamadigi icin sentetik uretiminden HARIC).
- Porsche Boxster: 2005 oncesi (986 nesli, 4 satir) vs 2005+ (987/981 nesli,
  4 satir) - capraz cift YOK.
- Mercedes-Benz GLS: tek kume (3 satir, hepsi ayni "GLS D" varyanti,
  2016-2017) - kisitlama gerekmiyor, SADECE 3 essiz cift var (zorlama yok).
- Audi Q4 Sportback: 288hp kumesi (4 satir) kullanilir, 213hp'lik TEK satir
  (farkli batarya/motor tier, es bulunamiyor) HARIC.
- Maserati GranTurismo: 4.2L kumesi (4 satir) ve 4.7L kumesi (9 satir) -
  capraz cift YOK (kullanicinin acik talebi).
- Audi Q8 Sportback E-Tron: tek kume (6 satir) - 408/413hp farki (~%1) gercek
  farkli powertrain DEGIL, olcum/trim varyansi - ayirmaya gerek yok.

Calistirma (ai-model/ calisma dizini olarak): python generate_third_wave_preview.py
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
from generate_synthetic_pilot import MAX_RETRIES_PER_ROW, _attempt_row, _source_label

THIRD_WAVE_SEED = 34567
GENERATED_AT = datetime.now(timezone.utc).isoformat()
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'synthetic_third_wave_preview.csv')

ANOMALY_EXCLUDE = {'arabam-40704969'}  # Audi TTS - 2250.5cc/463hp, standart TTS speklerine uymuyor

PLAN = [
    ('Audi', 'TTS', [
        ('mk2_2008_2012', ['kaggle-ab-15454', 'kaggle-ab-9403', 'arabam-41457812', 'kaggle-ab-37517',
                            'arabam-41743097', 'kaggle-ab-8014', 'kaggle-ab-21325']),
        ('mk3_2016_single_excluded', ['arabam-40086630']),  # n=1, es yok -> hicbir cift URETILMEZ
    ], 5),
    ('Porsche', 'Boxster', [
        ('gen_986_pre2005', ['arabam-40488741', 'arabam-42139768', 'kaggle-ab-2420', 'kaggle-ab-2392']),
        ('gen_987_981_2005plus', ['kaggle-ab-2402', 'arabam-37244743', 'kaggle-ab-2416', 'kaggle-ab-3847']),
    ], 5),
    ('Mercedes - Benz', 'GLS', [
        ('tek_kume_gls_d', None),
    ], 6),
    ('Audi', 'Q4 Sportback', [
        ('etron_288hp', ['arabam-41702587', 'arabam-41628212', 'arabam-41140561', 'arabam-41475659']),
        ('etron_213hp_single_excluded', ['arabam-42134577']),  # n=1, es yok
    ], 6),
    ('Maserati', 'GranTurismo', [
        ('4.2L', ['kaggle-ab-991', 'kaggle-ab-974', 'arabam-29455568', 'arabam-37773040']),
        ('4.7L', ['kaggle-ab-984', 'kaggle-ab-975', 'kaggle-ab-981', 'arabam-41561742', 'kaggle-ab-990',
                   'arabam-39045347', 'arabam-41697784', 'arabam-42005171', 'arabam-42176084']),
    ], 3),
    ('Audi', 'Q8 Sportback E-Tron', [
        ('tek_kume', None),
    ], 5),
]


def generate_clustered(marka, model, clusters, n_target, rng, hp_lookup, clean_df):
    full_group = clean_df[(clean_df['marka'] == marka) & (clean_df['model'] == model)].copy()
    full_group = full_group[~full_group['ilan_id'].isin(ANOMALY_EXCLUDE)]
    full_group = full_group.set_index('ilan_id')

    all_pairs = []
    excluded_singleton_clusters = []
    for cluster_adi, ids in clusters:
        if ids is None:
            ids = list(full_group.index)
        ids = [i for i in ids if i in full_group.index]
        if len(ids) < 2:
            excluded_singleton_clusters.append((cluster_adi, ids))
            continue
        for i_id, j_id in itertools.combinations(ids, 2):
            all_pairs.append((cluster_adi, i_id, j_id))

    if not all_pairs:
        return [], 0, excluded_singleton_clusters

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
            print(f'  UYARI: {marka} {model} [{cluster_adi}] cift ({i_id}+{j_id}) guvenli araliga girmedi, ATLANDI/REJECT')
            continue

        k += 1
        accepted['ilan_id'] = f'synthetic3-{marka}-{model}-{cluster_adi}-{k}'.replace(' ', '_')
        accepted['marka'] = marka
        accepted['model'] = model
        accepted['generation_cluster'] = cluster_adi
        accepted['donor_parent_id'] = accepted['categorical_donor_id']
        accepted['is_synthetic'] = 1
        accepted['source'] = 'synthetic_wave34_preview'
        accepted['synthetic_method'] = 'log_linear_2parent_interp(yil,km,fiyat)+single_donor_categoricals+bounded_noise+curve_guard+generation_cluster_restricted'
        accepted['synthetic_seed'] = THIRD_WAVE_SEED
        accepted['generated_at'] = GENERATED_AT
        accepted['synthetic_safe_check'] = True
        out.append(accepted)
    return out, len(all_pairs), excluded_singleton_clusters


def main():
    print('Production preprocessing ile TEMIZ (gercek) veri okunuyor (SADECE OKUMA)...')
    clean = load_clean_train_dataset()

    print('Production hierarchical_price artefakti okunuyor (SADECE OKUMA)...')
    import joblib
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'lightgbm_final.joblib'))
    hp_lookup = artifact['hierarchical_price']

    rng = np.random.default_rng(THIRD_WAVE_SEED)
    all_rows = []
    summary = []
    for marka, model, clusters, n_target in PLAN:
        real_n = len(clean[(clean['marka'] == marka) & (clean['model'] == model)])
        usable_n = real_n - sum(1 for c in clusters for i in (c[1] or []) if i in ANOMALY_EXCLUDE)
        print(f'\n{marka} {model}: gercek n={real_n}, hedef={n_target}, kume sayisi={len(clusters)}')
        generated, n_pairs, excluded_singles = generate_clustered(marka, model, clusters, n_target, rng, hp_lookup, clean)
        for c_adi, ids in excluded_singles:
            print(f'  NOT: kume [{c_adi}] tek satirli ({ids}) - es bulunamadigi icin HARIC (sentetik uretilmedi)')
        print(f'  essiz kume-ici cift sayisi: {n_pairs}, uretilen: {len(generated)}')
        if len(generated) < n_target:
            print(f'  NOT: hedef {n_target} idi, sadece {len(generated)} uretildi (kume kisitlamasi/curve guard reddi)')
        all_rows.extend(generated)
        summary.append({
            'model': f'{marka}|{model}', 'real_n': real_n, 'usable_real_n': usable_n,
            'target': n_target, 'generated': len(generated), 'n_clusters': len(clusters),
        })

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'\nYAZILDI (train_dataset.csv/wave30/wave31 CSV DEGISMEDI, retrain YAPILMADI): {os.path.abspath(OUT_PATH)}')
    print(f'Toplam sentetik satir: {len(df)} / hedef {sum(s["target"] for s in summary)}')

    # dogrulama: anomali satiri hicbir source_parent_ids'te yok mu
    if len(df) > 0:
        anomaly_hits = df['source_parent_ids'].str.contains('arabam-40704969', na=False).sum()
        print(f'\nDOGRULAMA: arabam-40704969 (TTS anomalisi) source_parent_ids icinde gecme sayisi: {anomaly_hits} (0 olmali)')

    print('\n=== MODEL OZETI ===')
    for s in summary:
        print(f"  {s['model']}: real_n={s['real_n']} usable={s['usable_real_n']} hedef={s['target']} uretilen={s['generated']} kume={s['n_clusters']}")


if __name__ == '__main__':
    main()
