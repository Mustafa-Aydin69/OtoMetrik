/**
 * sahibinden.com tarzı üst-seviye araç kategorisi (Otomobil / Arazi, SUV &
 * Pickup / Minivan & Panelvan / Elektrikli Araç) — bu bir eğitim-zamanı alanı
 * DEĞİLDİR (arac_turu preprocess.py'de tamamen atılıyor, bkz. CLAUDE.md).
 * Sadece VehicleSelector'ın ilk kademesi için Marka/Model listesini daraltan
 * bir ÖNFİLTREDİR — kanonik kasa_turu (Otomobil/Arazi&SUV&Pickup/Minivan&
 * Panelvan) veya yakıt türü (Elektrikli Araç, kasa tipinden bağımsız) üzerinden
 * BODY_TYPE_BY_MODEL / ENGINES_BY_MODEL'e (vehicle-options.generated.ts) karşı
 * türetilir — /predict'e hiçbir zaman gönderilmez.
 */
import { canonicalToLabel, labelToCanonical, vehicleKey } from "./validation";
import { BODY_TYPE_BY_MODEL, ENGINES_BY_MODEL, MODELS_BY_BRAND } from "./vehicle-options.generated";

export const VEHICLE_CATEGORIES = [
  "Otomobil",
  "Arazi, SUV & Pickup",
  "Minivan & Panelvan",
  "Elektrikli Araç",
] as const;

export type VehicleCategory = (typeof VEHICLE_CATEGORIES)[number];

export const CATEGORY_ICON: Record<VehicleCategory, string> = {
  Otomobil: "🚗",
  "Arazi, SUV & Pickup": "🚙",
  "Minivan & Panelvan": "🚐",
  "Elektrikli Araç": "🔌",
};

// Kanonik kasa_turu değerleri (bkz. category-options.generated.ts KASA_TURU_OPTIONS).
// "Elektrikli Araç" burada YOK - kasa tipinden bağımsız, ayrıca yakıt türü
// üzerinden ele alınır (bkz. modelMatchesCategory).
const CATEGORY_TO_KASA_TURU: Record<Exclude<VehicleCategory, "Elektrikli Araç">, readonly string[]> = {
  Otomobil: ["Sedan", "Hatchback/3", "Hatchback/5", "Coupe", "Station wagon", "Cabrio", "Roadster", "Hard top"],
  "Arazi, SUV & Pickup": ["SUV", "Pick-up", "Crossover"],
  "Minivan & Panelvan": ["MPV", "Panel Van", "Camlı Van", "Yarım Camlı Van", "Frigorifik Panelvan", "Minibüs"],
};

const trCollator = new Intl.Collator("tr", { numeric: true, sensitivity: "base" });

/** brandLabel/model çifti bu kategoriye giriyor mu - marka/model listelerini daraltmak için. */
export function modelMatchesCategory(canonicalBrand: string, model: string, category: VehicleCategory): boolean {
  const key = vehicleKey(canonicalBrand, model);
  if (category === "Elektrikli Araç") {
    const engines = ENGINES_BY_MODEL[key] ?? [];
    return engines.some((e) => e.yakitTuru === "Elektrik");
  }
  const kasaTurus = BODY_TYPE_BY_MODEL[key] ?? [];
  const allowed = CATEGORY_TO_KASA_TURU[category];
  return kasaTurus.some((k) => allowed.includes(k));
}

/** Bu kategoride en az bir modeli olan markaların etiketlerini tr-collator sırasına göre döner. */
export function getBrandLabelsForCategory(category: VehicleCategory): string[] {
  const labels: string[] = [];
  for (const [canonicalBrand, models] of Object.entries(MODELS_BY_BRAND)) {
    if (models.some((m) => modelMatchesCategory(canonicalBrand, m, category))) {
      labels.push(canonicalToLabel("brand", canonicalBrand));
    }
  }
  return labels.sort(trCollator.compare);
}

/** Seçilen marka için, bu kategoriye giren modellerin etiketlerini tr-collator sırasına göre döner. */
export function getModelLabelsForBrandAndCategory(brandLabel: string, category: VehicleCategory): string[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  const models = MODELS_BY_BRAND[canonicalBrand] ?? [];
  return models.filter((m) => modelMatchesCategory(canonicalBrand, m, category)).sort(trCollator.compare);
}
