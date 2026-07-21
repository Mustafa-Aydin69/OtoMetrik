/**
 * NormalizedVehicle'dan arama motoru/Commons için metin sorguları üretir.
 * En zenginden en genele doğru birden fazla varyant döner — image-provider.ts
 * ilkinin sonuç vermemesi durumunda bir sonrakine düşer (eksik alan varsa
 * çıkarılmış, daha geniş bir sorgu).
 */
import type { NormalizedVehicle } from "./types";

function join(parts: Array<string | null | undefined>): string {
  return parts
    .filter((p): p is string => !!p && p.trim().length > 0)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * En spesifikten en genele sıralı benzersiz sorgu varyantları:
 * [marka+model+nesil+paket+renk+yıl, ...renk/paket'i sırayla düşürerek..., marka+model]
 */
export function buildQueryVariants(vehicle: NormalizedVehicle): string[] {
  const { brand, model, generation, trim, color, year } = vehicle;
  const yearStr = year ? String(year) : null;

  const variants = [
    join([brand, model, generation, trim, color, yearStr]),
    join([brand, model, generation, color, yearStr]),
    join([brand, model, generation, trim]),
    join([brand, model, generation]),
    join([brand, model, yearStr]),
    join([brand, model]),
  ];

  const seen = new Set<string>();
  const unique: string[] = [];
  for (const v of variants) {
    if (v && !seen.has(v)) {
      seen.add(v);
      unique.push(v);
    }
  }
  return unique;
}

/** Sıralama (image-ranker.ts) için sorgudaki anlamlı kelimeler. */
export function queryTokens(vehicle: NormalizedVehicle): string[] {
  const raw = [
    vehicle.brand,
    vehicle.model,
    vehicle.generation,
    vehicle.trim,
    vehicle.color,
    vehicle.year ? String(vehicle.year) : null,
  ];
  return raw
    .filter((p): p is string => !!p && p.trim().length > 0)
    .flatMap((p) => p.split(/\s+/))
    .map((t) => t.trim())
    .filter(Boolean);
}
