/**
 * Model adını Wikipedia/Commons'ın anladığı İngilizce forma çevirir - "Vehicle
 * Identity Resolution" katmanı. İki ayrı mantık var, KARIŞTIRILMAMALI:
 *
 * 1. Generic automotive normalization (MODEL_TERM_EN): marka-spesifik OLMAYAN
 *    Türkçe otomotiv nomenklatür kelimeleri -> İngilizce ("Serisi" -> "Series").
 *    COLOR_EN (vehicle-parser.ts) ile aynı desen.
 *
 * 2. Brand-specific strategy (BRAND_MODEL_TERM_STRATEGY): bazı markalarda
 *    generic çeviri YANLIŞ sonuç üretir - canlı Wikipedia testiyle doğrulandı:
 *
 *      "BMW 3 Serisi"          -> "BMW 3 Series"   (TRANSLATE doğru, redirect var)
 *      "Mercedes-Benz V Serisi" -> "Mercedes-Benz V Series" YOK (missingtitle)
 *                                  ama "Mercedes-Benz V" VAR (Wikipedia'nın
 *                                  kendi redirect'i zaten "V"yi tanıyor)
 *      "Audi 100 Serisi"       -> "Audi 100 Series" YOK, "Audi 100" VAR
 *
 *    Yani Mercedes-Benz ve Audi'de "Serisi" kelimesi ÇEVRİLMEMELİ, SİLİNMELİDİR
 *    (STRIP) - bare model kodu zaten Wikipedia'nın kendi redirect sisteminde
 *    tanınıyor (örn. "Mercedes-Benz C" -> "Mercedes-Benz C-Class" otomatik
 *    çözülüyor, action=parse&redirects=1 ile canlı doğrulandı). Bu, marka
 *    başına devasa bir model kataloğu DEĞİL - sadece iki markanın "Serisi"
 *    kelimesine nasıl davrandığını belirleyen tek satırlık bir strateji
 *    seçimi. Mevcut veri setinde (vehicle-options.generated.ts) "Serisi"
 *    kelimesini kullanan markalar: BMW, Audi, Mercedes-Benz, Ford, Maserati -
 *    Ford ("E Serisi" -> "Ford E Series" -> Wikipedia kendi redirect'iyle
 *    "Ford E-Series"e yönlendiriyor) ve Maserati (nadir/şüpheli veri) default
 *    (TRANSLATE) stratejisiyle sorun çıkarmıyor, bu yüzden yalnızca
 *    Mercedes-Benz + Audi için override yeterli.
 */

const MODEL_TERM_EN: Record<string, string> = {
  serisi: "Series",
  sınıfı: "Class",
  sınıf: "Class",
};

type ModelTermStrategy = "translate" | "strip";

// Anahtar: form'un/kullanıcının gördüğü marka ETİKETİ (category-options.generated.ts
// MARKA_OPTIONS[].label) - image pipeline'a giden vehicle.brand bu, dataset'in
// kanonik değeri ("Mercedes - Benz", boşluklu) DEĞİL.
const BRAND_MODEL_TERM_STRATEGY: Record<string, ModelTermStrategy> = {
  "Mercedes-Benz": "strip",
  Audi: "strip",
};

/**
 * Model adındaki generic otomotiv terimlerini markaya göre çevirir ya da siler.
 * Terim sözlükte yoksa kelime olduğu gibi kalır (marka/model kataloğu
 * OLUŞTURMAYA çalışmaz - yalnızca bilinen birkaç nomenklatür kelimesini ele alır).
 */
export function canonicalizeModelText(brand: string, model: string): string {
  const strategy = BRAND_MODEL_TERM_STRATEGY[brand] ?? "translate";
  return model
    .split(/\s+/)
    .map((word) => {
      const key = word.toLocaleLowerCase("tr");
      const translated = MODEL_TERM_EN[key];
      if (translated === undefined) return word;
      return strategy === "strip" ? null : translated;
    })
    .filter((word): word is string => word !== null && word.length > 0)
    .join(" ");
}
