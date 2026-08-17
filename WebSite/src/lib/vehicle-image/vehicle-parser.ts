/**
 * Araç bilgisini pipeline'ın ortak VehicleIdentity şekline çevirir - "Vehicle
 * Normalization" katmanı (generation alanları burada BOŞ kalır, onları
 * generation-resolver.ts doldurur, bkz. vehicle-image.service.ts).
 *
 * İki giriş yolu var:
 * - normalizeVehicleInput: birincil/güvenilir yol. Site formu zaten
 *   yapılandırılmış alanlar (brand/model/year/trim/color/bodyType) sağlıyor -
 *   burada gerçek bir "ayrıştırma" yok, sadece: renk çevirisi, model
 *   kanonikleştirme (bkz. vehicle-aliases.ts) ve trim metninden variant
 *   (motor/alt-model kodu, örn. "320i") ayrıştırma.
 * - parseFreeTextVehicle: "2010 gri Ford Fiesta Titanium" gibi serbest
 *   metinden en iyi çaba (best-effort) çıkarım yapar. Marka/model/paket
 *   ayrımı hardcoded bir marka/model kataloğu olmadan güvenilir şekilde
 *   yapılamaz (bu belirsizlik LLM ile çözülür) - bu yüzden bu fonksiyon
 *   sadece yıl ve rengi güvenle çıkarır, geri kalan kelimeleri "rest" olarak
 *   bırakır (çağıran, rest'i brand/model/trim'e nasıl dağıtacağına kendi
 *   bağlamıyla karar verir). Site şu an bu yolu kullanmıyor (form zaten
 *   yapılandırılmış); ileride serbest metin girişi eklenirse temel taşı.
 *
 * ÖNEMLİ SINIR: Buradaki normalizasyon (örn. "3 Serisi" -> "3 Series") YALNIZCA
 * görsel arama alanı içindir. Fiyat tahmin modeline (/api/predict) giden
 * PredictionInput'a HİÇ dokunmaz - o ayrı bir pipeline, ayrı bir şema
 * (bkz. lib/validation.ts). Modelin beklediği kategorik değerleri
 * (örn. "3 Serisi") burada değiştirmek prediction'ı bozar.
 */
import { canonicalizeModelText } from "./vehicle-aliases";
import type { VehicleIdentity } from "./types";

// WebSite'ın kendi COLORS listesindeki (src/lib/validation.ts) sabit Türkçe
// değerleri İngilizce'ye çevirir - Wikipedia/Commons/Google aramaları
// İngilizce metin üzerinde çok daha iyi sonuç verir.
const COLOR_EN: Record<string, string> = {
  siyah: "black",
  beyaz: "white",
  gri: "grey",
  gümüş: "silver",
  mavi: "blue",
  kırmızı: "red",
  kahverengi: "brown",
  bej: "beige",
  yeşil: "green",
  turuncu: "orange",
  sarı: "yellow",
  bordo: "maroon",
  lacivert: "navy",
};

// Zaten İngilizce/başka dilde girilmiş renkler için normalize edilmiş
// (lowercase) kimlik eşlemesi - yabancı veri kaynaklarıyla uyum için.
const KNOWN_ENGLISH_COLORS = new Set([
  "black", "white", "grey", "gray", "silver", "blue", "red", "brown",
  "beige", "green", "orange", "yellow", "maroon", "navy", "gold", "purple",
]);

function translateColor(color: string | null | undefined): string | null {
  if (!color) return null;
  const key = color.trim().toLowerCase();
  if (!key) return null;
  return COLOR_EN[key] ?? (KNOWN_ENGLISH_COLORS.has(key) ? key : null);
}

// Sitedeki paket/trim metni ham scrape verisi - bazen "{motor kodu} {trim}"
// (örn. "320i Executive M Sport"), bazen sadece trim (örn. "Comfort") ya da
// eksik/tutarsız (örn. "i Comfort" - baştaki rakam kayıp, bkz. plan
// bulgusu). Yalnızca AÇIKÇA bir motor kodu görünümündeki (2-3 rakam +
// 1-4 harf) baştaki token'ı variant olarak ayırır; eşleşmezse metnin
// TAMAMI trim olarak korunur - yanlış bir ayrıştırma üretmektense hiç
// ayrıştırmamak tercih edilir (bkz. plan Madde 4).
const VARIANT_PREFIX = /^(\d{2,3}[a-zA-Z]{1,4})\s+(.+)$/;

function extractVariant(trimText: string | null): { variant: string | null; trim: string | null } {
  if (!trimText) return { variant: null, trim: null };
  const match = trimText.match(VARIANT_PREFIX);
  if (!match) return { variant: null, trim: trimText };
  return { variant: match[1], trim: match[2].trim() || null };
}

export interface VehicleFormInput {
  brand: string;
  model: string;
  year?: number | null;
  trim?: string | null;
  bodyType?: string | null;
  color?: string | null;
}

/** Birincil yol: form'un zaten yapılandırılmış alanlarından VehicleIdentity üretir (generation alanları boş). */
export function normalizeVehicleInput(input: VehicleFormInput): VehicleIdentity {
  const make = input.brand.trim();
  const rawModel = input.model.trim();
  const { variant, trim } = extractVariant(input.trim?.trim() || null);

  return {
    make,
    model: canonicalizeModelText(make, rawModel),
    rawModel,
    variant,
    trim,
    year: typeof input.year === "number" && Number.isFinite(input.year) ? input.year : null,
    bodyType: input.bodyType?.trim() || null,
    color: translateColor(input.color),
    generation: null,
    generationOrdinalLabel: null,
    facelift: null,
    generationStartYear: null,
    generationEndYear: null,
    generationSource: "unknown",
  };
}

export interface FreeTextParseResult {
  year: number | null;
  color: string | null;
  /** Yıl ve renk çıkarıldıktan sonra kalan kelimeler (marka/model/paket karışık). */
  rest: string;
}

const TR_COLOR_WORDS = Object.keys(COLOR_EN);

// Best-effort: hardcoded marka/model kataloğu olmadan bir cümleden marka,
// model ve paketi güvenilir biçimde ayırmak mümkün değil (bu, doğası gereği
// LLM gerektiren bir belirsizlik gidermedir). Bu fonksiyon sadece net
// biçimde tespit edilebilen yıl ve rengi çıkarır.
export function parseFreeTextVehicle(text: string): FreeTextParseResult {
  let rest = text.trim();
  let year: number | null = null;
  let color: string | null = null;

  const yearMatch = rest.match(/\b(19[5-9]\d|20[0-4]\d)\b/);
  if (yearMatch) {
    year = Number(yearMatch[1]);
    rest = rest.replace(yearMatch[0], " ");
  }

  for (const trWord of TR_COLOR_WORDS) {
    const re = new RegExp(`\\b${trWord}\\b`, "i");
    if (re.test(rest)) {
      color = COLOR_EN[trWord];
      rest = rest.replace(re, " ");
      break;
    }
  }
  if (!color) {
    for (const enWord of KNOWN_ENGLISH_COLORS) {
      const re = new RegExp(`\\b${enWord}\\b`, "i");
      if (re.test(rest)) {
        color = enWord;
        rest = rest.replace(re, " ");
        break;
      }
    }
  }

  rest = rest.replace(/\s+/g, " ").trim();
  return { year, color, rest };
}
