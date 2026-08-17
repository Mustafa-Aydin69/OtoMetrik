/**
 * Hard Validation/Conflict Filtering + Metadata Scoring katmanı - eski
 * image-ranker.ts'in "sadece lexical overlap" mantığının yerine geçer
 * (bkz. plan Madde 9-13).
 *
 * İki AYRI mekanizma var, KARIŞTIRILMAMALI:
 *
 * 1. Hard conflict rejection (detectHardConflict): adayın metadata'sı AÇIKÇA
 *    başka bir nesil/model/yıl ifade ediyorsa -1 yerine DOĞRUDAN reddedilir
 *    (score düşürmek yetmez, bkz. plan Madde 10 - "target G20, candidate E46
 *    -> REJECT"). Generation conflict tespiti, generation-rules.ts'teki
 *    KENDİ marka+model için bilinen KOD LİSTESİYLE sınırlıdır (yalnızca
 *    "E46 vs G20" gibi doğrulanmış kod çiftleri) - rastgele bir token'ın
 *    "başka bir nesil" sanılıp yanlış reddedilmesi (false positive) riskini
 *    en aza indirir. Model conflict tespiti ise BMW "N Series" / Mercedes
 *    "X-Class" gibi dar, spesifik kalıplarla sınırlıdır (bkz. plan Madde 11 -
 *    "C/E/S gibi tek harf/tek rakam tokenlar kullanma").
 *
 * 2. Weighted scoring (scoreCandidate): reddedilmeyen adaylar için, hangi
 *    kimlik alanlarının metadata'da doğrulandığına göre ağırlıklı puan -
 *    generation en yüksek ağırlık, renk en düşük (bkz. plan Madde 13 -
 *    "generation correctness renkten çok daha önemlidir").
 */
import { GENERATION_RULES } from "./generation-rules";
import type { ImageCandidate, MatchedFields, RankedImageCandidate, VehicleIdentity } from "./types";

const WEIGHTS = {
  make: 25,
  model: 30,
  generation: 40,
  year: 20,
  bodyType: 10,
  variant: 10,
  trim: 8,
  color: 4,
} as const;

/** confidence.ts'in skoru 0-1'e normalize etmek için kullandığı, tüm ağırlıkların toplamı. */
export const MAX_SCORE = Object.values(WEIGHTS).reduce((sum, w) => sum + w, 0);

/** confidence.ts'in zero-overlap güvenlik ağı için de kullandığı, adayın tüm metadata'sını birleştiren metin. */
export function searchableText(c: ImageCandidate): string {
  return [c.title, c.filename, c.description, ...(c.categories ?? [])]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function includesToken(text: string, token: string | null): boolean {
  if (!token) return false;
  return text.includes(token.toLowerCase());
}

/** trim genelde çok kelimeli ("Executive M Sport") - tam ifade nadiren birebir geçer, tek anlamlı kelime yeterli sayılır. */
function includesAnyWord(text: string, phrase: string | null): boolean {
  if (!phrase) return false;
  return phrase
    .split(/\s+/)
    .filter((w) => w.length >= 3)
    .some((w) => text.includes(w.toLowerCase()));
}

// Nesil kodu şekli: 1-2 harf + 1-3 rakam (ör. E46, F30, G20, W204) VEYA
// rakam + 1-2 harf (ör. 8V, 8Y) - hem harf hem rakam ZORUNLU, tek başına
// harf ("C","E","S" gibi Mercedes model kodlarıyla çakışır) ya da tek
// başına rakam asla eşleşmez (bkz. plan Madde 11 - false positive riski).
const GENERATION_CODE_SHAPE = /^(?=[a-z0-9]{2,4}$)(?=.*[a-z])(?=.*\d)[a-z0-9]+$/i;

function extractCodeShapedTokens(text: string): string[] {
  return (text.match(/[a-z0-9]{2,5}/gi) ?? []).filter((t) => GENERATION_CODE_SHAPE.test(t));
}

/**
 * target'ın marka+model'i için generation-rules.ts'te bilinen TÜM kodlar
 * (target'ın kendi nesli HARİÇ) - adayın metninde bunlardan biri açıkça
 * geçiyorsa bu güvenilir bir "başka nesil" sinyalidir.
 */
function knownAlternateGenerationCodes(vehicle: VehicleIdentity): Set<string> {
  const codes = GENERATION_RULES.filter((r) => r.make === vehicle.make && r.model === vehicle.model).map((r) =>
    r.generation.toLowerCase()
  );
  return new Set(codes.filter((c) => c !== vehicle.generation?.toLowerCase()));
}

// BMW "N Series" / Mercedes "X-Class" gibi marka-spesifik model-ifadesi
// kalıpları - target'ın model NUMARASI/HARFİNDEN farklısı adayda açıkça
// geçiyorsa model conflict sayılır. Dar ve spesifik tutulur (section 11'in
// "devasa marka/model kataloğu yazma" uyarısına uygun).
function detectModelConflict(text: string, vehicle: VehicleIdentity): string | null {
  const bmwSeriesMatch = vehicle.model.match(/^(\d)\s*Series$/i);
  if (vehicle.make === "BMW" && bmwSeriesMatch) {
    const targetNum = bmwSeriesMatch[1];
    const found = [...text.matchAll(/\b(\d)\s*series\b/gi)].map((m) => m[1]);
    if (found.some((n) => n !== targetNum)) {
      return `model conflict: candidate metni "${found.find((n) => n !== targetNum)} Series" içeriyor, hedef "${targetNum} Series"`;
    }
  }

  const mercedesClassMatch = vehicle.model.match(/^([A-Z]{1,3})$/);
  if (vehicle.make === "Mercedes-Benz" && mercedesClassMatch) {
    const targetLetter = mercedesClassMatch[1].toLowerCase();
    const found = [...text.matchAll(/\b([a-z]{1,3})-class\b/gi)].map((m) => m[1].toLowerCase());
    if (found.some((l) => l !== targetLetter)) {
      return `model conflict: candidate metni "${found.find((l) => l !== targetLetter)}-Class" içeriyor, hedef "${targetLetter.toUpperCase()}-Class"`;
    }
  }

  return null;
}

function extractYearsFromText(text: string): number[] {
  return [...text.matchAll(/\b(19[5-9]\d|20[0-4]\d)\b/g)].map((m) => Number(m[1]));
}

/**
 * true/false: adayda bulunan yıl(lar) hedefle uyumlu mu. "unknown": adayda
 * hiç yıl bulunamadı (bkz. plan Madde 12 - "yıl yazmıyor" != "yanlış yıl",
 * sadece unknown, ret sebebi DEĞİL).
 */
function yearCompatibility(text: string, vehicle: VehicleIdentity): boolean | "unknown" {
  const found = extractYearsFromText(text);
  if (found.length === 0) return "unknown";
  if (vehicle.generationStartYear !== null) {
    return found.some(
      (y) => y >= vehicle.generationStartYear! && (vehicle.generationEndYear === null || y <= vehicle.generationEndYear!)
    );
  }
  if (vehicle.year !== null) return found.includes(vehicle.year);
  return "unknown";
}

function detectHardConflict(text: string, vehicle: VehicleIdentity): string | null {
  if (vehicle.generation) {
    const alternates = knownAlternateGenerationCodes(vehicle);
    if (alternates.size > 0) {
      const found = extractCodeShapedTokens(text).map((t) => t.toLowerCase());
      const conflictingCode = found.find((t) => alternates.has(t));
      if (conflictingCode) {
        return `generation conflict: candidate "${conflictingCode.toUpperCase()}" içeriyor, hedef "${vehicle.generation}"`;
      }
    }
  }

  const modelConflict = detectModelConflict(text, vehicle);
  if (modelConflict) return modelConflict;

  if (yearCompatibility(text, vehicle) === false) {
    return "year conflict: candidate metnindeki yıl(lar) hedef nesil aralığıyla/yılla uyumsuz";
  }

  return null;
}

function scoreCandidate(candidate: ImageCandidate, vehicle: VehicleIdentity): RankedImageCandidate {
  const text = searchableText(candidate);

  const rejectionReason = detectHardConflict(text, vehicle);
  if (rejectionReason) {
    const matched: MatchedFields = {
      make: includesToken(text, vehicle.make),
      model: includesToken(text, vehicle.model),
      generation: false,
      year: "unknown",
      variant: includesToken(text, vehicle.variant),
      trim: includesAnyWord(text, vehicle.trim),
      bodyType: includesToken(text, vehicle.bodyType),
      color: includesToken(text, vehicle.color),
    };
    return { ...candidate, score: 0, matched, rejected: true, rejectionReason };
  }

  const yearMatch = yearCompatibility(text, vehicle);
  const matched: MatchedFields = {
    make: includesToken(text, vehicle.make),
    model: includesToken(text, vehicle.model),
    generation: includesToken(text, vehicle.generation),
    year: yearMatch,
    variant: includesToken(text, vehicle.variant),
    trim: includesAnyWord(text, vehicle.trim),
    bodyType: includesToken(text, vehicle.bodyType),
    color: includesToken(text, vehicle.color),
  };

  let score = 0;
  if (matched.make) score += WEIGHTS.make;
  if (matched.model) score += WEIGHTS.model;
  if (matched.generation) score += WEIGHTS.generation;
  if (matched.year === true) score += WEIGHTS.year;
  if (matched.bodyType) score += WEIGHTS.bodyType;
  if (matched.variant) score += WEIGHTS.variant;
  if (matched.trim) score += WEIGHTS.trim;
  if (matched.color) score += WEIGHTS.color;

  return { ...candidate, score, matched, rejected: false };
}

/** Adayları değerlendirir: hard-reject uygulanmışlar `rejected:true` ile işaretlenir (elenmez, debug için tutulur), kalanlar skora göre azalan sıralanır. */
export function rankCandidates(candidates: ImageCandidate[], vehicle: VehicleIdentity): RankedImageCandidate[] {
  return candidates
    .map((c) => scoreCandidate(c, vehicle))
    .sort((a, b) => {
      if (a.rejected !== b.rejected) return a.rejected ? 1 : -1;
      return b.score - a.score;
    });
}
