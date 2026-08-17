/**
 * VehicleIdentity'den arama motoru/Commons için öncelik merdiveni (query
 * ladder) üretir - eski query-builder.ts'in "tek string, soldan sağa alan
 * sil" fallback'inin yerine geçer (bkz. plan Madde 7).
 *
 * Temel prensip: bilgi kaybı önceliği yaklaşık olarak
 *   color < trim < variant < exact year < bodyType < facelift < generation < model < make
 * (en önce color düşer, en son make düşer). generation BİLİNİYORSA ladder
 * onu ASLA bırakmaz - en düşürülmüş rung bile "make + generation (+facelift)"
 * seviyesinde kalır, generation'sız "make + model" seviyesine hiç inmez
 * (bkz. plan Madde 17 - "3 Series" seviyesine inildiğinde artık generation
 * doğrulanamıyorsa yanlış nesil riski geri döner). generation BİLİNMİYORSA
 * (bu bilgi hiç yoksa) normal merdiven model/make'e kadar iner - bu durumda
 * candidate-ranker.ts + confidence.ts düşük güveni zaten yakalar.
 */
import type { VehicleIdentity } from "./types";

export const MAX_QUERIES = 6;

function joinTokens(parts: Array<string | null | undefined>): string {
  return parts
    .filter((p): p is string => !!p && p.trim().length > 0)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function dedupeCapped(queries: string[], limit: number): string[] {
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const q of queries) {
    if (q && !seen.has(q)) {
      seen.add(q);
      unique.push(q);
    }
    if (unique.length >= limit) break;
  }
  return unique;
}

export function generateQueries(vehicle: VehicleIdentity): string[] {
  const { make, model, variant, trim, year, bodyType, color, generation, facelift } = vehicle;
  const yearStr = year ? String(year) : null;
  // En spesifik "kimlik" token'ı: variant biliniyorsa o (örn. "320i"),
  // bilinmiyorsa model (örn. "3 Series") - ikisini birden tekrar etmeye gerek yok.
  const identity = variant ?? model;

  const rungs: string[] = generation
    ? [
        joinTokens([yearStr, make, identity, generation, facelift, trim, bodyType, color]),
        joinTokens([yearStr, make, identity, generation, facelift, trim, bodyType]),
        joinTokens([yearStr, make, identity, generation, facelift, trim]),
        joinTokens([make, identity, generation, facelift]),
        joinTokens([make, model, generation, facelift]),
        joinTokens([make, generation, facelift]),
      ]
    : [
        joinTokens([yearStr, make, identity, trim, bodyType, color]),
        joinTokens([yearStr, make, identity, trim, bodyType]),
        joinTokens([yearStr, make, identity, trim]),
        joinTokens([make, identity]),
        joinTokens([make, model]),
        make,
      ];

  return dedupeCapped(rungs, MAX_QUERIES);
}

/** Sıralama (candidate-ranker.ts) için sorgudaki anlamlı kelimeler - her alan tek tek token'lara bölünür. */
export function identityTokens(vehicle: VehicleIdentity): string[] {
  const raw = [
    vehicle.make,
    vehicle.model,
    vehicle.variant,
    vehicle.generation,
    vehicle.facelift,
    vehicle.trim,
    vehicle.bodyType,
    vehicle.color,
    vehicle.year ? String(vehicle.year) : null,
  ];
  return raw
    .filter((p): p is string => !!p && p.trim().length > 0)
    .flatMap((p) => p.split(/\s+/))
    .map((t) => t.trim())
    .filter(Boolean);
}
