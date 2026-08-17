/**
 * Yerel/deterministik nesil bilgisi - generation-resolver.ts bunu Wikipedia'ya
 * gitmeden ÖNCE dener (bkz. o dosyanın katmanlı resolveGeneration() akışı).
 * Wikipedia SPOF (single point of failure) olmasın diye var - bu liste
 * KASITLI olarak küçük ve tam otomotiv tarihini kapsamıyor: yalnızca
 * yaygın/iyi belgelenmiş, üzerinde neredeyse hiç ihtilaf olmayan nesil kodları
 * (E46, G20, W204 gibi - araç meraklısı camiasında standart referanslar)
 * içeriyor. Emin olunamayan bir nesil eklemektense HİÇ eklememek tercih
 * edildi (bkz. plan: "yanlış facelift/nesil bilgisi, hiç bilgi olmamasından
 * daha kötü").
 *
 * `model` alanı, vehicle-aliases.ts'in kanonikleştirdiği HALİYLE eşleşmeli
 * (örn. BMW için "3 Series" - "Serisi" çevrilir; Mercedes-Benz için bare "C" -
 * "Serisi" yoksa zaten değişmez, "C-Class" DEĞİL, bkz. vehicle-aliases.ts
 * docstring'i).
 *
 * Yeni satır eklerken: yalnızca başlangıç yılından eminsen ekle, endYear'ı
 * BOŞ bırakabilirsin (bir sonraki neslin startYear'ından türetilir, bkz.
 * generation-resolver.ts resolveLocalGeneration()). facelift yalnızca kesin
 * biliniyorsa doldurulur.
 */
import type { GenerationRule } from "./types";

export const GENERATION_RULES: GenerationRule[] = [
  // BMW 3 Series
  { make: "BMW", model: "3 Series", generation: "E36", ordinalLabel: "Third generation", startYear: 1990 },
  { make: "BMW", model: "3 Series", generation: "E46", ordinalLabel: "Fourth generation", startYear: 1998 },
  { make: "BMW", model: "3 Series", generation: "E90", ordinalLabel: "Fifth generation", startYear: 2005 },
  { make: "BMW", model: "3 Series", generation: "F30", ordinalLabel: "Sixth generation", startYear: 2011 },
  {
    make: "BMW",
    model: "3 Series",
    generation: "G20",
    ordinalLabel: "Seventh generation",
    startYear: 2018,
    facelift: "LCI",
    faceliftStartYear: 2022,
  },

  // BMW 5 Series
  { make: "BMW", model: "5 Series", generation: "E34", ordinalLabel: "Third generation", startYear: 1988 },
  { make: "BMW", model: "5 Series", generation: "E39", ordinalLabel: "Fourth generation", startYear: 1995 },
  { make: "BMW", model: "5 Series", generation: "E60", ordinalLabel: "Fifth generation", startYear: 2003 },
  { make: "BMW", model: "5 Series", generation: "F10", ordinalLabel: "Sixth generation", startYear: 2010 },
  { make: "BMW", model: "5 Series", generation: "G30", ordinalLabel: "Seventh generation", startYear: 2016 },

  // Mercedes-Benz C (dataset model kodu bare "C" - bkz. vehicle-aliases.ts)
  { make: "Mercedes-Benz", model: "C", generation: "W202", ordinalLabel: "First generation", startYear: 1993 },
  { make: "Mercedes-Benz", model: "C", generation: "W203", ordinalLabel: "Second generation", startYear: 2000 },
  { make: "Mercedes-Benz", model: "C", generation: "W204", ordinalLabel: "Third generation", startYear: 2007 },
  { make: "Mercedes-Benz", model: "C", generation: "W205", ordinalLabel: "Fourth generation", startYear: 2014 },
  { make: "Mercedes-Benz", model: "C", generation: "W206", ordinalLabel: "Fifth generation", startYear: 2021 },

  // Mercedes-Benz E
  { make: "Mercedes-Benz", model: "E", generation: "W124", ordinalLabel: "First generation", startYear: 1984 },
  { make: "Mercedes-Benz", model: "E", generation: "W210", ordinalLabel: "Second generation", startYear: 1995 },
  { make: "Mercedes-Benz", model: "E", generation: "W211", ordinalLabel: "Third generation", startYear: 2002 },
  { make: "Mercedes-Benz", model: "E", generation: "W212", ordinalLabel: "Fourth generation", startYear: 2009 },
  { make: "Mercedes-Benz", model: "E", generation: "W213", ordinalLabel: "Fifth generation", startYear: 2016 },
];

/** Yıl verilmemişse veya bilinen bir kural yoksa null döner - resolver Wikipedia'ya düşer. */
export function resolveLocalGeneration(make: string, model: string, year: number | null): GenerationRule | null {
  if (!year) return null;
  const rules = GENERATION_RULES.filter((r) => r.make === make && r.model === model).sort(
    (a, b) => a.startYear - b.startYear
  );
  let match: GenerationRule | null = null;
  for (const rule of rules) {
    if (rule.startYear <= year) match = rule;
    else break;
  }
  if (!match) return null;
  if (match.endYear !== undefined && year > match.endYear) return null;
  return match;
}

/** match'in KENDİSİNDEN SONRAKİ kuralın startYear'ı - endYear yoksa üst sınır türetmek için. */
export function nextRuleStartYear(rule: GenerationRule): number | null {
  const rules = GENERATION_RULES.filter((r) => r.make === rule.make && r.model === rule.model).sort(
    (a, b) => a.startYear - b.startYear
  );
  const idx = rules.findIndex((r) => r.generation === rule.generation);
  return idx >= 0 && idx + 1 < rules.length ? rules[idx + 1].startYear : null;
}
