"""Faz 18: WebSite/src/lib/paket-suggestions.generated.ts'yi egitim verisinden
uretir. Faz 11 SHAP + Faz 15 error-taxonomy bulgusu: paket'in dusuk SHAP payi
(~%5.6 gain) VE frekans-hata korelasyonunun zayifligi (model_ratio 2.92x vs
paket_ratio 1.05x, bkz. hata taksonomisi raporu) gosterdi ki asil sorun MODEL
DOGRULUGU degil - kullanicinin serbest metinle yazdigi paket'in egitim
kelime hazinesiyle EŞLEŞMEMESI (paket'lerin %47'si egitimde frekans<=5).

category_mapping.py'den (Faz 13) FARKLI bir yaklasim gerekir: paket icin
tek bir kucuk label->canonical sozlugu YAZILAMAZ (6.444 farkli deger).
Bunun yerine, kullaniciyi mevcut kelime hazinesine YONLENDIREN bir
otomatik-tamamlama listesi uretilir - marka+model secilince o modelin
GERCEKTEN egitimde gorulen paket degerleri (frekansa gore siralanmis)
onerilir. Kullanici oneriden birini secerse gonderilen deger zaten
kanoniktir (translate katmani GEREKMEZ, cunku paket zaten scrape edilmis
ham string'in kendisidir - Faz 13'teki "Manuel"/"Duz" gibi iki farkli
vokabuler sorunu burada yok).

Calistirma (ai-model/ calisma dizini olarak): python generate_paket_suggestions.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json

from category_mapping import LABEL_TO_CANONICAL
from train import prepare_full_training_data

WEBSITE_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'WebSite', 'src', 'lib', 'paket-suggestions.generated.ts'
)

HEADER = """/**
 * OTOMATIK URETILMISTIR - ELLE DUZENLEMEYIN.
 *
 * Kaynak: WebScrape/ai-model/generate_paket_suggestions.py (egitim verisindeki
 * gercek marka+model+paket kombinasyonlari, frekansa gore siralanmis).
 * Uretmek icin: cd WebScrape/ai-model && python generate_paket_suggestions.py
 *
 * Anahtar: "<kanonik marka>|<model>" (bkz. category-options.generated.ts
 * MARKA_OPTIONS'daki value alanlari - "Mercedes - Benz" gibi kanonik marka
 * degerleri kullanilir, website etiketi ("Mercedes-Benz") DEGIL).
 * Deger: o marka+model icin egitimde gorulen paket degerleri, en sik
 * kullanilandan en nadire dogru siralanmis.
 */

export const PAKET_BY_MODEL: Record<string, string[]> = """


def build_paket_by_model(X_full):
    grouped = X_full.groupby(['marka', 'model'], observed=True)['paket']
    result = {}
    for (marka, model), group in grouped:
        # paket kategorik dtype - value_counts() varsayilan olarak dtype'daki
        # TUM kategorileri (bu grupta HIC gorulmemis olanlar dahil, sayim=0)
        # dondurur. counts>0 filtresi olmadan her grup GLOBAL 6.444 paketin
        # tamamini "oneri" olarak alirdi (ilk denemede 5.1 MILYON oneri
        # uretmisti - acik bir hata).
        counts = group.value_counts()
        counts = counts[counts > 0]
        pakets = [str(p) for p in counts.index if str(p) != 'nan']
        if pakets:
            result[f'{marka}|{model}'] = pakets
    return result


def main():
    X_full, y_full = prepare_full_training_data()
    paket_by_model = build_paket_by_model(X_full)

    # saglik kontrolu: butun kanonik marka degerleri (category_mapping.py'nin
    # ASIL kaynagi) burada da tutarli mi - drift erken yakalanir.
    canonical_markas = {v for v in LABEL_TO_CANONICAL['marka'].values() if v}
    seen_markas = {key.split('|', 1)[0] for key in paket_by_model}
    unknown = seen_markas - canonical_markas
    if unknown:
        print(f'UYARI: egitim verisinde olup category_mapping.py kanonik listesinde '
              f'olmayan {len(unknown)} marka var (website dropdown\'unda YOK, bu '
              f'modeller icin paket onerisi hicbir zaman tetiklenmez): {sorted(unknown)[:10]}...')

    total_models = len(paket_by_model)
    total_pakets = sum(len(v) for v in paket_by_model.values())
    print(f'{total_models} marka+model grubu, toplam {total_pakets} paket onerisi uretildi.')

    output_path = os.path.abspath(WEBSITE_OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(HEADER)
        f.write(json.dumps(paket_by_model, ensure_ascii=False, indent=2))
        f.write(';\n')
    print(f'uretildi: {output_path}')


if __name__ == '__main__':
    main()
