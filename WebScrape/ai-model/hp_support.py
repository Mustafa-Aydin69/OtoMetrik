"""Faz 17: motor_gucu icin peer-support tabanli guven (confidence) mekanizmasi.

Faz 16 (hp_quality_analysis.py) yuksek guc segmentindeki zayifligin ANA
NEDENININ veri kirliligi degil GERCEK ORNEKLEM YETERSIZLIGI oldugunu
gosterdi (400+ HP: sadece 170 egitim kaydi, R2=-3.18). Bu modul, tahmini
REDDETMEK yerine "bu tahmin ne kadar guvenilir" bilgisini serve.py
uzerinden ACIKCA iletir - dagitima hazirlik analizindeki Madde 6 tasarim
onerisinin (fiziksel sinir / istatistiksel uc deger / guvenilir destek
bolgesi ayrimi) UYGULAMASIDIR.

Performans (Madde 6 - request basina DataFrame taramasi YASAK): tum ozetler
build_support_summary() ile EGITIM ZAMANINDA hesaplanip model artefaktina
gomulur (train.py save_model()). serve.py sadece kucuk dict/bisect
aramalari yapar - O(1)/O(log n), DataFrame'e hic dokunmaz.

Bin genisligi (Madde 1 - "HP dilimleri ... model ici dagilima gore
tanimlansin"): marka+model grubu icin bin genisligi O GRUBUN KENDI MAD'inden
(1.4826*MAD, min 10 HP taban) turetilir - yogun/dar dagilimli modellerde dar,
gevsek/genis dagilimli modellerde genis dilimler olusur. SADECE marka+model
ornek sayisi yetersizse (MIN_MODEL_COUNT_FOR_PRIMARY altinda) SABIT genislikli
(FALLBACK_BIN_WIDTH=50 HP) marka-seviyesi (veya son care global) fallback'e
dusulur - "sabit dilim yalnizca fallback" talebiyle birebir uyumlu.

Guven esikleri (Madde 2) - korlemesine degil, Faz 15 error-taxonomy'deki
GERCEK model-frekans/hata iliskisinden turetildi: orada model frekansi
1-5 -> MAE 208k (3x kotu), 6-20 -> 96k, 21-100 -> 76k, 101-500 -> 68k
(baseline'a en yakin) bulunmustu - yani ~20-50 ornek civari MAE'nin
"baseline'a yakinsadigi" gecis bandidir; bu modulun "medium/high" esikleri
(10-50) bu gecisle uyumlu secildi.
"""
import numpy as np

MIN_MODEL_COUNT_FOR_PRIMARY = 20  # bu altinda marka+model bin'leri guvenilmez, marka'ya duser
FALLBACK_BIN_WIDTH = 50.0  # HP - marka/global fallback icin SABIT genislik
MIN_MAD_BIN_WIDTH = 10.0  # HP - marka+model bin genisligi icin taban (MAD~0 durumunda dejenere binlemeyi onler)

PERCENTILE_POINTS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]

# bkz. modul docstring'i - Faz 15 model-frekans/MAE iliskisinden turetildi.
HIGH_PEER_COUNT_THRESHOLD = 50
MEDIUM_PEER_COUNT_THRESHOLD = 10
MIN_MODEL_COUNT_FOR_MEDIUM = 20  # model'in KENDISI de bu kadar ornek icermeli, yoksa "medium" "low"a duser


def _group_summary(hp_series, bin_width):
    n = len(hp_series)
    if n == 0:
        return None
    percentiles = {p: float(np.percentile(hp_series, p)) for p in PERCENTILE_POINTS}
    bin_idx = np.floor(hp_series / bin_width).astype(int)
    bin_counts = bin_idx.value_counts().to_dict()
    return {
        'count': int(n),
        'percentiles': percentiles,
        'bin_width': float(bin_width),
        'bin_counts': {int(k): int(v) for k, v in bin_counts.items()},
    }


def build_support_summary(X_full):
    """X_full: train.py prepare_full_training_data()'nin dondurdugu, motor_gucu/
    marka/model sutunlarini iceren egitim ozellik dataframe'i. Model artefaktina
    'hp_support' anahtariyla gomulmek uzere joblib-serilestirilebilir bir dict
    dondurur."""
    # Faz 30: preprocess.py artik motor_gucu NaN'i satiri dusurmeden birakiyor
    # (bkz. preprocess.py Faz 30 notu) - NaN bir HP degeri tasimadigi icin bu
    # ozetin (HP dagilim/peer-support) hesabina hic giremez, burada ayri dropna.
    df = X_full[['marka', 'model', 'motor_gucu']].dropna(subset=['motor_gucu']).copy()
    df['marka'] = df['marka'].astype(str)
    df['model'] = df['model'].astype(str)

    model_stats = {}
    for (marka, model), g in df.groupby(['marka', 'model'], observed=True):
        hp = g['motor_gucu']
        if len(hp) < 2:
            continue
        med = hp.median()
        mad = (hp - med).abs().median()
        bin_width = max(1.4826 * mad, MIN_MAD_BIN_WIDTH)
        summary = _group_summary(hp, bin_width)
        if summary is not None:
            model_stats[f'{marka}\x1f{model}'] = summary

    marka_stats = {}
    for marka, g in df.groupby('marka', observed=True):
        summary = _group_summary(g['motor_gucu'], FALLBACK_BIN_WIDTH)
        if summary is not None:
            marka_stats[str(marka)] = summary

    global_stats = _group_summary(df['motor_gucu'], FALLBACK_BIN_WIDTH)

    return {
        'model_stats': model_stats,
        'marka_stats': marka_stats,
        'global_stats': global_stats,
        'min_model_count_for_primary': MIN_MODEL_COUNT_FOR_PRIMARY,
    }


def _percentile_rank(hp, percentiles):
    """percentiles: {1:.., 5:.., ..., 99.9:..} - hp'nin bu noktalar arasinda
    DOGRUSAL interpolasyonla tahmini yuzdelik siralamasi (raw veriye erisim
    olmadan, sadece onceden hesaplanmis noktalardan)."""
    points = sorted(percentiles.items())
    xs = [v for _, v in points]
    ys = [p for p, _ in points]
    if hp <= xs[0]:
        return ys[0] if hp == xs[0] else max(0.0, ys[0] - 1)
    if hp >= xs[-1]:
        return ys[-1] if hp == xs[-1] else min(100.0, ys[-1] + (100 - ys[-1]) * 0.5)
    return float(np.interp(hp, xs, ys))


def _peer_count_from_bins(hp, bin_width, bin_counts):
    idx = int(hp // bin_width)
    return bin_counts.get(idx, 0) + bin_counts.get(idx - 1, 0) + bin_counts.get(idx + 1, 0)


def lookup_support(marka, model, hp, support):
    """(peer_count, model_count, hp_percentile, peer_group) dondurur.
    Once marka+model (yeterli ornekse), sonra marka, sonra global fallback."""
    key = f'{marka}\x1f{model}'
    model_entry = support['model_stats'].get(key)
    if model_entry is not None and model_entry['count'] >= support.get(
            'min_model_count_for_primary', MIN_MODEL_COUNT_FOR_PRIMARY):
        peer_count = _peer_count_from_bins(hp, model_entry['bin_width'], model_entry['bin_counts'])
        pct = _percentile_rank(hp, model_entry['percentiles'])
        return peer_count, model_entry['count'], pct, 'marka+model'

    marka_entry = support['marka_stats'].get(marka)
    if marka_entry is not None and marka_entry['count'] > 0:
        peer_count = _peer_count_from_bins(hp, marka_entry['bin_width'], marka_entry['bin_counts'])
        pct = _percentile_rank(hp, marka_entry['percentiles'])
        model_count = model_entry['count'] if model_entry is not None else 0
        return peer_count, model_count, pct, 'marka (fallback)'

    global_entry = support['global_stats']
    peer_count = _peer_count_from_bins(hp, global_entry['bin_width'], global_entry['bin_counts'])
    pct = _percentile_rank(hp, global_entry['percentiles'])
    model_count = model_entry['count'] if model_entry is not None else 0
    return peer_count, model_count, pct, 'global (fallback)'


def compute_confidence(peer_count, model_count, peer_group):
    """bkz. modul docstring'i - esikler Faz 15'teki gercek frekans/hata
    iliskisinden turetildi. hp_percentile'in [P5,P95]/[P1,P99] disinda olmasi
    zaten DUSUK peer_count'a yansir (bin'de az ornek -> dusuk peer_count),
    bu yuzden ayri bir "aralik disinda" kontrolüne GEREK YOK - bin-tabanli
    peer_count tek basina hem yerel yogunlugu hem uc-deger durumunu yakalar.

    peer_group KRITIK bir girdi: marka+model YETERSIZ ornekliyse (lookup_support
    marka/global'e DUSTUYSE), peer_count artik marka/global CAPINDA bir yogunluk
    olcer - "bu SPESIFIK model+HP kombinasyonu" hakkinda BILGI TASIMAZ (orn.
    5 ornekli nadir bir modelin markasi 7000 ornekli olabilir - bu, o nadir
    model icin YUKSEK guven ANLAMINA GELMEZ). Bu yuzden fallback'e dusulduyse
    guven 'medium' ile SINIRLANIR - "sadece HP'ye degil MODEL+HP kombinasyonuna
    bak" talebini karsilamak icin gereklidir (bkz. gorev talebi Madde 2)."""
    if peer_group != 'marka+model':
        return 'medium' if model_count >= MEDIUM_PEER_COUNT_THRESHOLD else 'low'
    if peer_count >= HIGH_PEER_COUNT_THRESHOLD:
        return 'high'
    if peer_count >= MEDIUM_PEER_COUNT_THRESHOLD:
        return 'medium'
    return 'low'
