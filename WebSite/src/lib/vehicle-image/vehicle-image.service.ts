/**
 * Araç görsel bulma pipeline'ının orkestrasyon katmanı — diğer tüm
 * modülleri sırayla çağırır:
 *
 *   1. vehicle-parser      : form girdisini NormalizedVehicle'a çevirir
 *   2. generation-resolver  : brand+model+year'dan Wikipedia'ya bakıp nesli çözer
 *   3. query-builder        : en spesifikten en genele sorgu varyantları üretir
 *   4. image-provider       : Commons -> (opsiyonel Google CSE) -> Wikipedia infobox
 *   5. image-ranker         : adayları sorguyla örtüşmesine göre sıralar
 *
 * Dışa açılan tek fonksiyon getVehicleImages — api/car-image/route.ts bunu
 * çağırır ve sıralı URL listesini (en iyi eşleşme önce) client'a döner.
 */
import { resolveGeneration } from "./generation-resolver";
import { getImageCandidates } from "./image-provider";
import { lexicalRanker } from "./image-ranker";
import { buildQueryVariants, queryTokens } from "./query-builder";
import { normalizeVehicleInput, type VehicleFormInput } from "./vehicle-parser";

export async function getVehicleImages(input: VehicleFormInput, limit = 10): Promise<string[]> {
  const base = normalizeVehicleInput(input);
  if (!base.brand || !base.model) return [];

  const generation = await resolveGeneration(base.brand, base.model, base.year);
  const vehicle = { ...base, generation: generation?.label ?? null };

  const variants = buildQueryVariants(vehicle);
  if (variants.length === 0) return [];

  const candidates = await getImageCandidates(vehicle.brand, vehicle.model, variants, limit);
  if (candidates.length === 0) return [];

  const tokens = queryTokens(vehicle);
  // Nesil çözülebildiyse sıra ifadesini de (örn. "Seventh generation")
  // arama anahtar kelimelerine ekle — bazı Commons açıklamaları kısa kodu
  // değil bu ifadeyi içerir.
  if (generation?.ordinalLabel) tokens.push(...generation.ordinalLabel.split(/\s+/));

  const ranked = lexicalRanker.rank(candidates, tokens);
  return ranked.map((c) => c.url);
}
