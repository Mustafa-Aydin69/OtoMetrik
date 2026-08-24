"""Faz 25: WebSite/src/lib/vehicle-options.generated.ts'yi egitim verisinden
uretir. generate_paket_suggestions.py'nin (Faz 18) yerine gecer - o script
yalnizca marka+model -> paket ureten TEK katmanli bir yapiydi; kart tabanli
Marka > Model > Motor > Paket akisi icin araya bir Motor (motor_hacmi +
yakit_turu) katmani gerekiyor, paket de artik marka+model DEGIL marka+model+
motor bazinda anlamli (bkz. kart tasarim referansi - ayni modelin farkli
motorlarinda gercekten farkli paket kombinasyonlari goruluyor).

Motor gruplama karari (bu Faz'da olculdu): motor_hacmi HAM haliyle
gruplanirsa (orn. Mazda 3) marka+model basina medyan 5, bazi modellerde
(Mercedes-Benz E) 79 farkli deger cikiyor - bunlarin cogu GERCEK farkli
motorlar DEGIL, prepare_train_dataset.py'deki "range-text -> orta nokta"
normalizasyonundan kaynaklanan neredeyse-ayni float gurultusu (1496/1499/
1500.0/1500.5 gibi). motor_hacmi'yi EN YAKIN 100cc'ye yuvarlamak (sadece
GORUNTULEME/GRUPLAMA icin) grup sayisini 6616'dan 4832'ye, en kotu modeli
79'dan 47'ye indiriyor. Kullaniciya /predict'e GONDERILEN engineDisplacement
degeri ise o kovadaki EN SIK GORULEN (mode) gercek cc degeri - kova sinirinin
kendisi (orn. 1600) degil, cunku model kovayi degil gercek sayiyi gordu.

Motor Gucu (HP) coklugu de bu Faz'da olculdu: kovalanmis motor gruplarinin
%42.9'unda (2074/4832) birden fazla farkli motor_gucu degeri var (ort. 1.95,
maks. 22) - website'in "birden fazla gecerli HP varsa secilebilir goster,
tekse otomatik doldur" davranisi spekulatif degil, verinin gercek bir ozelligi.

Faz 26: kullanici Citroen Berlingo 1.5 Dizel + Shine Bold icin motor_gucu
dropdown'inda 6 deger gordu (96/100/102/110/130/132) ama gercek dunyada bu
motorun yalnizca ~100 ve ~130 HP olmak uzere iki gercek versiyonu var - 102 ve
132 gibi degerler muhtemelen DIN/olcum yuvarlama farki (motor_hacmi'deki
"range-text -> orta nokta" gurultusune benzer bir veri kalitesi sorunu).
motor_hacmi'deki AYNI cozum uygulanir: motor_gucu EN YAKIN 5 HP'ye yuvarlanir
(GORUNTULEME/GRUPLAMA icin), her kova icin o kovadaki EN SIK GORULEN gercek HP
degeri secilebilir/gonderilebilir deger olarak kullanilir. Berlingo 1.5 Dizel
ornegi: 6 ham deger -> 4 kova (95, 100, 110, 130) - 95 ve 110 tek-satirlik
nadir aykiri degerler olarak ayri kalmaya devam ediyor (kasitli - bunlari
gercek verinin bir parcasi olarak filtrelemek ayri bir veri temizligi karari
gerektirir, bu Faz'in kapsami degil).

Faz 26 ayrica: kasa_turu (Kasa Tipi) artik marka+model'e gore GERCEK gorulen
degerlerle sinirlaniyor (BODY_TYPE_BY_MODEL) - onceden website'deki 11 sabit
secenek her araca ayni sekilde sunuluyordu; Citroen Berlingo gibi ticari
araclarda gercek kasa tipi ("Camlı Van", satirlarinin %87'si) o listede hic
yoktu, kullanici "Sedan" gibi o araca gercekte hic uymayan bir deger secmek
zorunda kaliyordu (bkz. category_mapping.py'nin Faz 26 notu - kasa_turu artik
17 kanonik degerin tamamini kapsiyor).

Faz 27: motor (ENGINES_BY_MODEL) ve motor gucu (HP_BY_ENGINE) gruplarina
MIN_GROUP_COUNT esigi eklendi - Mercedes-Benz C icin "1.5 Elektrik" (count=1)
kartinin website'de Benzin/Dizel/Hibrit gibi gercek secenklerle AYNI gorunurlukte
sunuldugu tespit edildi (Mercedes hic tam elektrikli C serisi uretmedi; bu
satir muhtemelen bir mild-hybrid C 200 AMG ilaninin yanlis etiketlenmesi).
Faz 26'da HP kovalari icin "tek-satirlik nadir aykiri degerler kasitli olarak
filtrelenmiyor" denmisti - bu karar artik gecerli DEGIL: count < MIN_GROUP_COUNT
olan hem motor hem HP kovalari eleniyor (bkz. build_vehicle_options).

Calistirma (ai-model/ calisma dizini olarak): python generate_vehicle_options.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json

from category_mapping import LABEL_TO_CANONICAL
from train import prepare_full_training_data

WEBSITE_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'WebSite', 'src', 'lib', 'vehicle-options.generated.ts'
)

# Bu esigin ALTINDAKI (yani tek satirlik) marka+model+motor / motor+HP
# kombinasyonlari veri girisi hatasi supheli sayilip elenir (bkz. Faz 27 notu).
MIN_GROUP_COUNT = 2

HEADER = """/**
 * OTOMATIK URETILMISTIR - ELLE DUZENLEMEYIN.
 *
 * Kaynak: WebScrape/ai-model/generate_vehicle_options.py (egitim verisindeki
 * gercek marka+model+motor+paket kombinasyonlari).
 * Uretmek icin: cd WebScrape/ai-model && python generate_vehicle_options.py
 *
 * Kart tabanli Marka > Model > Motor > Paket akisinin TEK veri kaynagi.
 * Anahtar formati: "<kanonik marka>|<model>" (MODELS_BY_BRAND haric, o
 * dogrudan marka'ya gore anahtarlanir) ve motor-bazli haritalarda
 * "<kanonik marka>|<model>|<hacmiBucket>|<yakitTuru>".
 *
 * hacmiBucket: motor_hacmi'nin en yakin 100cc'ye yuvarlanmis hali (GORUNTULEME/
 * GRUPLAMA icin - orn. 1600). exactCc (ENGINES_BY_MODEL icinde) o kovadaki
 * EN SIK GORULEN gercek motor_hacmi degeri - /predict'e GONDERILMESI gereken
 * budur, hacmiBucket'in kendisi degil.
 */

export interface EngineOption {
  hacmiBucket: number;
  yakitTuru: string;
  exactCc: number;
  count: number;
}
"""


def build_body_types_by_model(X_full):
    d = X_full.dropna(subset=['kasa_turu'])
    result = {}
    for (marka, model), group in d.groupby(['marka', 'model'], observed=True):
        counts = group['kasa_turu'].value_counts()
        counts = counts[counts > 0]
        values = [str(v) for v in counts.index if str(v) != 'nan']
        if values:
            result[f'{marka}|{model}'] = values
    return result


def build_vehicle_options(X_full):
    d = X_full.dropna(subset=['motor_hacmi', 'yakit_turu']).copy()
    d['hacmi_bucket'] = (d['motor_hacmi'] / 100).round() * 100

    models_by_brand = {}
    for (marka, model), _ in X_full.groupby(['marka', 'model'], observed=True):
        models_by_brand.setdefault(str(marka), []).append(str(model))

    engines_by_model = {}
    for (marka, model, bucket, yakit), group in d.groupby(
        ['marka', 'model', 'hacmi_bucket', 'yakit_turu'], observed=True
    ):
        if len(group) < MIN_GROUP_COUNT:
            continue
        key = f'{marka}|{model}'
        exact_cc = float(group['motor_hacmi'].mode().iloc[0])
        engines_by_model.setdefault(key, []).append({
            'hacmiBucket': float(bucket),
            'yakitTuru': str(yakit),
            'exactCc': exact_cc,
            'count': int(len(group)),
        })
    for key in engines_by_model:
        engines_by_model[key].sort(key=lambda e: (e['hacmiBucket'], e['yakitTuru']))

    def engine_key(marka, model, bucket, yakit):
        return f'{marka}|{model}|{bucket}|{yakit}'

    paket_by_engine = {}
    for (marka, model, bucket, yakit, paket), group in d.groupby(
        ['marka', 'model', 'hacmi_bucket', 'yakit_turu', 'paket'], observed=True
    ):
        if not group.size:
            continue
        key = engine_key(marka, model, bucket, yakit)
        paket_by_engine.setdefault(key, []).append((str(paket), len(group)))
    paket_by_engine = {
        key: [p for p, _ in sorted(items, key=lambda pc: pc[1], reverse=True)]
        for key, items in paket_by_engine.items()
        if items
    }
    # kategori dtype grupla(observed=True) 'nan' stringini de bir kategori
    # olarak dondurebiliyor - gercek paket olmayan bu deger temizlenir (bkz.
    # generate_paket_suggestions.py'deki ayni sorun/cozum).
    for key, pakets in list(paket_by_engine.items()):
        cleaned = [p for p in pakets if p != 'nan']
        if cleaned:
            paket_by_engine[key] = cleaned
        else:
            del paket_by_engine[key]

    # motor_gucu de motor_hacmi ile ayni sorunu tasiyor: 100/102, 130/132 gibi
    # birbirine 2 HP mesafede degerler ayni gercek versiyonun olcum/yuvarlama
    # gurultusu (bkz. modul docstring'i, Faz 26). En yakin 5 HP'ye yuvarlanip
    # her kova icin EN SIK GORULEN gercek motor_gucu degeri secilebilir/
    # gonderilebilir deger olarak alinir - motor_hacmi->exactCc ile ayni desen.
    hp_by_engine = {}
    for (marka, model, bucket, yakit), group in d.dropna(subset=['motor_gucu']).groupby(
        ['marka', 'model', 'hacmi_bucket', 'yakit_turu'], observed=True
    ):
        key = engine_key(marka, model, bucket, yakit)
        hp = group['motor_gucu'].dropna().copy()
        hp_bucket = (hp / 5).round() * 5
        representative_values = set()
        for b in hp_bucket.unique():
            bucket_values = hp[hp_bucket == b]
            if len(bucket_values) < MIN_GROUP_COUNT:
                continue
            mode_val = bucket_values.mode().iloc[0]
            representative_values.add(float(mode_val))
        if representative_values:
            hp_by_engine[key] = sorted(representative_values)

    return models_by_brand, engines_by_model, paket_by_engine, hp_by_engine


def main():
    X_full, y_full = prepare_full_training_data()
    models_by_brand, engines_by_model, paket_by_engine, hp_by_engine = build_vehicle_options(X_full)
    body_types_by_model = build_body_types_by_model(X_full)

    # saglik kontrolu: category_mapping.py'nin kanonik marka listesinde
    # OLMAYAN bir marka egitim verisinde varsa erken uyar (drift).
    canonical_markas = {v for v in LABEL_TO_CANONICAL['marka'].values() if v}
    seen_markas = set(models_by_brand.keys())
    unknown = seen_markas - canonical_markas
    if unknown:
        print(f'UYARI: egitim verisinde olup category_mapping.py kanonik listesinde '
              f'olmayan {len(unknown)} marka var (website VehicleSelector\'unda hic '
              f'gorunmez): {sorted(unknown)[:10]}...')

    total_models = sum(len(v) for v in models_by_brand.values())
    total_engine_groups = sum(len(v) for v in engines_by_model.values())
    print(f'{len(models_by_brand)} marka, {total_models} marka+model grubu')
    print(f'{total_engine_groups} kovalanmis marka+model+motor grubu')
    print(f'{len(paket_by_engine)} marka+model+motor grubu icin paket onerisi')
    print(f'{len(hp_by_engine)} marka+model+motor grubu icin motor gucu secenegi')
    multi_hp = sum(1 for v in hp_by_engine.values() if len(v) > 1)
    print(f'  bunlarin {multi_hp}\'i ({100 * multi_hp / len(hp_by_engine):.1f}%) birden fazla HP degerine sahip')
    print(f'{len(body_types_by_model)} marka+model grubu icin kasa tipi secenegi')
    multi_body = sum(1 for v in body_types_by_model.values() if len(v) > 1)
    print(f'  bunlarin {multi_body}\'i ({100 * multi_body / len(body_types_by_model):.1f}%) birden fazla kasa tipine sahip')

    output_path = os.path.abspath(WEBSITE_OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(HEADER)
        f.write('\nexport const MODELS_BY_BRAND: Record<string, string[]> = ')
        f.write(json.dumps(models_by_brand, ensure_ascii=False, indent=2))
        f.write(';\n')
        f.write('\nexport const ENGINES_BY_MODEL: Record<string, EngineOption[]> = ')
        f.write(json.dumps(engines_by_model, ensure_ascii=False, indent=2))
        f.write(';\n')
        f.write('\nexport const PAKET_BY_ENGINE: Record<string, string[]> = ')
        f.write(json.dumps(paket_by_engine, ensure_ascii=False, indent=2))
        f.write(';\n')
        f.write('\nexport const HP_BY_ENGINE: Record<string, number[]> = ')
        f.write(json.dumps(hp_by_engine, ensure_ascii=False, indent=2))
        f.write(';\n')
        f.write('\nexport const BODY_TYPE_BY_MODEL: Record<string, string[]> = ')
        f.write(json.dumps(body_types_by_model, ensure_ascii=False, indent=2))
        f.write(';\n')
    print(f'uretildi: {output_path}')


if __name__ == '__main__':
    main()
