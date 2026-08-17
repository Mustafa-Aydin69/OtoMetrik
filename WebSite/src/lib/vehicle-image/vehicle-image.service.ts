/**
 * Araç görsel bulma pipeline'ının orkestrasyon katmanı - diğer tüm
 * modülleri sırayla çağırır (bkz. plan Madde 2 - 9 aşamalı akış):
 *
 *   1. vehicle-parser      : form girdisini VehicleIdentity'ye çevirir (Vehicle Normalization)
 *   2. vehicle-aliases      : model adını kanonikleştirir (Vehicle Identity Resolution, vehicle-parser içinde çağrılır)
 *   3. generation-resolver  : local rules -> Wikipedia (Generation/Facelift Resolution)
 *   4. query-generator      : öncelik merdiveni (Search Query Generation)
 *   5. providers            : Commons + Google CSE, rung rung dener (Candidate Retrieval)
 *   6. candidate-ranker     : hard-reject + ağırlıklı skor (Hard Validation + Metadata Scoring)
 *   7. confidence           : eşik + zero-overlap güvenlik ağı (Confidence Gate)
 *
 * Dışa açılan tek fonksiyon getVehicleImages - api/car-image/route.ts bunu
 * çağırır ve güven eşiğini geçen URL listesini (en iyi eşleşme önce) client'a
 * döner. Hiçbir aşama exception FIRLATMAZ - herhangi bir dış API sorunu
 * (timeout/429/malformed response) sessizce boş sonuca düşer, fiyat tahmin
 * akışını etkilemez (bkz. plan Madde 28 - "image enrichment pipeline
 * price prediction'dan gevşek bağlı olmalı").
 */
import { rankCandidates } from "./candidate-ranker";
import { selectConfidentImages } from "./confidence";
import { debugLog } from "./debug-log";
import { resolveGeneration } from "./generation-resolver";
import { fetchWikipediaInfobox, PROVIDERS } from "./providers";
import { generateQueries } from "./query-generator";
import type { ImageCandidate, RankedImageCandidate, VehicleIdentity } from "./types";
import { normalizeVehicleInput, type VehicleFormInput } from "./vehicle-parser";
import type { ConfidentSelection } from "./confidence";

// Tüm pipeline için üst sınır (bkz. plan Madde 29) - bireysel fetch'ler zaten
// kendi AbortSignal.timeout'larına sahip, ama rung'lar SIRAYLA denendiği için
// (erken çıkış performansı korumak amacıyla, bkz. resolveCandidates) en kötü
// durumda bunların toplamı fiyat tahminini gereksiz geciktirebilir. Bu süre
// aşılırsa boş sonuç dönülür - görsel özelliği opsiyoneldir, tahmini asla
// bloklamamalı.
const PIPELINE_TIMEOUT_MS = 12_000;

function dedupeCandidates(candidates: ImageCandidate[]): ImageCandidate[] {
  const seen = new Map<string, ImageCandidate>();
  for (const c of candidates) {
    const key = c.filename ?? c.url;
    if (!seen.has(key)) seen.set(key, c);
  }
  return [...seen.values()];
}

interface ResolvedCandidates {
  candidates: ImageCandidate[];
  ranked: RankedImageCandidate[];
  selection: ConfidentSelection;
  usedQuery: string | null;
}

/**
 * Query ladder'ı SIRAYLA dener (her rung'da tüm provider'lar paralel) - ilk
 * rung'da confidence eşiğini geçen bir aday bulunursa hemen durur (Madde 29 -
 * "ilk yüksek confidence candidate bulunduğunda erken çıkılabilir"). Bir
 * rung sonuç verip hiçbiri confidence eşiğini geçemezse (arama başarılı ama
 * hiçbir aday güvenilir değil) sonraki, daha genel rung'a devam eder - bu,
 * "arama başarısızlığı" ile "kimlik çözümü başarısızlığı"nı ayrı tutar
 * (Madde 18): kimlik (vehicle.generation vb.) sabit kalır, yalnızca sorgu
 * genişler.
 */
async function resolveCandidates(vehicle: VehicleIdentity, queries: string[], limit: number): Promise<ResolvedCandidates> {
  let last: ResolvedCandidates = {
    candidates: [],
    ranked: [],
    selection: { urls: [], best: { imageUrl: null, confidence: 0, rejectedCount: 0 } },
    usedQuery: null,
  };

  for (const query of queries) {
    const results = await Promise.all(PROVIDERS.map((p) => p.search(query, limit)));
    const combined = dedupeCandidates(results.flat());
    if (combined.length === 0) continue;

    const ranked = rankCandidates(combined, vehicle);
    const selection = selectConfidentImages(ranked, vehicle);
    last = { candidates: combined, ranked, selection, usedQuery: query };
    if (selection.urls.length > 0) break;
  }

  return last;
}

async function resolveVehicleImagesInner(input: VehicleFormInput, limit: number): Promise<string[]> {
  const base = normalizeVehicleInput(input);
  if (!base.make || !base.model) return [];

  const resolvedGeneration = await resolveGeneration(base.make, base.model, base.year);
  const vehicle: VehicleIdentity = resolvedGeneration
    ? {
        ...base,
        generation: resolvedGeneration.label,
        generationOrdinalLabel: resolvedGeneration.ordinalLabel,
        facelift: resolvedGeneration.facelift,
        generationStartYear: resolvedGeneration.startYear,
        generationEndYear: resolvedGeneration.endYear,
        generationSource: resolvedGeneration.source,
      }
    : base;

  const queries = generateQueries(vehicle);
  if (queries.length === 0) return [];

  let result = await resolveCandidates(vehicle, queries, limit);

  // Hiçbir rung confidence eşiğini geçen bir aday üretmediyse son çare:
  // Wikipedia infobox (tek aday, nesil/renk/paket filtrelemez - bu yüzden
  // yalnızca buraya kadar hiçbir güvenilir aday bulunamadıysa denenir).
  if (result.selection.urls.length === 0) {
    const infobox = await fetchWikipediaInfobox(vehicle.make, vehicle.model);
    if (infobox.length > 0) {
      const ranked = rankCandidates(infobox, vehicle);
      const selection = selectConfidentImages(ranked, vehicle);
      result = { candidates: infobox, ranked, selection, usedQuery: result.usedQuery };
    }
  }

  debugLog("resolution", {
    rawVehicle: input,
    normalizedVehicle: {
      make: vehicle.make,
      model: vehicle.model,
      rawModel: vehicle.rawModel,
      variant: vehicle.variant,
      trim: vehicle.trim,
      year: vehicle.year,
      bodyType: vehicle.bodyType,
      color: vehicle.color,
    },
    resolvedGeneration: {
      generation: vehicle.generation,
      facelift: vehicle.facelift,
      source: vehicle.generationSource,
      yearRange: [vehicle.generationStartYear, vehicle.generationEndYear],
    },
    queries,
    usedQuery: result.usedQuery,
    candidateCount: result.candidates.length,
    rejectedCandidates: result.ranked
      .filter((c) => c.rejected)
      .map((c) => ({ title: c.title ?? c.filename ?? c.url, reason: c.rejectionReason })),
    selected: result.selection.best,
  });

  return result.selection.urls;
}

export async function getVehicleImages(input: VehicleFormInput, limit = 10): Promise<string[]> {
  try {
    return await Promise.race([
      resolveVehicleImagesInner(input, limit),
      new Promise<string[]>((resolve) => setTimeout(() => resolve([]), PIPELINE_TIMEOUT_MS)),
    ]);
  } catch (err) {
    debugLog("error", { message: err instanceof Error ? err.message : String(err) });
    return [];
  }
}
