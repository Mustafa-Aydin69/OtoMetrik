/**
 * Vehicle-image pipeline robustness/regression audit script - ÜRETİM KODUNA
 * DOKUNMAZ, yalnızca src/lib/vehicle-image/*'teki mevcut fonksiyonları
 * (normalizeVehicleInput, resolveGeneration, generateQueries, PROVIDERS,
 * fetchWikipediaInfobox, rankCandidates, selectConfidentImages) tam olarak
 * vehicle-image.service.ts'in orkestrasyonuyla AYNI sırada çağırır - tek
 * fark, her aşamanın ARA çıktısını (debugLog'un aksine) bir JSON/CSV
 * dosyasına yazması, böylece audit sonrası analiz edilebilir.
 *
 * Çalıştırma: npx tsx scripts/vehicle-image-audit.ts
 * Çıktı: scripts/audit-results/audit-{timestamp}.json ve .csv
 *
 * Rate limit'e girmemek için CONCURRENCY sınırlı tutulur; mevcut cache.ts
 * (generation-resolver.ts + providers.ts tarafından zaten kullanılıyor)
 * aynı sorgunun tekrar ağa gitmesini zaten engelliyor.
 */
import { writeFileSync } from "node:fs";
import { rankCandidates } from "../src/lib/vehicle-image/candidate-ranker";
import { selectConfidentImages } from "../src/lib/vehicle-image/confidence";
import { resolveGeneration } from "../src/lib/vehicle-image/generation-resolver";
import { fetchWikipediaInfobox, PROVIDERS } from "../src/lib/vehicle-image/providers";
import { generateQueries } from "../src/lib/vehicle-image/query-generator";
import type { ImageCandidate, RankedImageCandidate, VehicleIdentity } from "../src/lib/vehicle-image/types";
import { normalizeVehicleInput } from "../src/lib/vehicle-image/vehicle-parser";
import { CASES, type TestCase } from "./audit-cases";

// İlk İKİ audit denemesinde (concurrency=4 ve concurrency=2+400ms) TÜM/NEREDEYSE
// TÜM vakalar candidateCount=0 döndü - izole kısa testler (8-10 istek) sorunsuz
// çalışırken, birkaç dakika süren TOPLAM istek hacmi Commons'ı rate-limit'e
// sokuyor (bkz. audit raporu "P0: rate limiting"). Bu üçüncü denemede
// concurrency=1 (tamamen seri) ve vakalar arası 2.5sn bekleme var - toplam
// çalışma süresi uzun (~10-15dk) ama istek YOĞUNLUĞU çok daha düşük.
const CONCURRENCY = 1;
const DELAY_BETWEEN_CASES_MS = 2500;

interface AuditResult {
  id: string;
  category: string;
  rawInput: TestCase;
  normalized: {
    make: string;
    model: string;
    rawModel: string;
    variant: string | null;
    trim: string | null;
    year: number | null;
    bodyType: string | null;
    color: string | null;
  };
  resolvedGeneration: string | null;
  resolvedFacelift: string | null;
  generationSource: "local" | "wikipedia" | "unknown";
  generationYearRange: [number | null, number | null];
  queries: string[];
  usedQuery: string | null;
  candidateCount: number;
  rejectedCount: number;
  rejectedReasons: Array<{ title: string; reason: string | undefined }>;
  selected: {
    url: string | null;
    title: string | null;
    source: string | null;
    score: number | null;
    confidence: number;
    matched: RankedImageCandidate["matched"] | null;
  };
  fromInfoboxFallback: boolean;
  error: string | null;
}

function dedupeCandidates(candidates: ImageCandidate[]): ImageCandidate[] {
  const seen = new Map<string, ImageCandidate>();
  for (const c of candidates) {
    const key = c.filename ?? c.url;
    if (!seen.has(key)) seen.set(key, c);
  }
  return [...seen.values()];
}

async function auditOne(tc: TestCase): Promise<AuditResult> {
  try {
    const base = normalizeVehicleInput(tc);
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
    if (resolvedGeneration?.source === "wikipedia") {
      await new Promise((r) => setTimeout(r, 500));
    }

    let candidates: ImageCandidate[] = [];
    let ranked: RankedImageCandidate[] = [];
    let usedQuery: string | null = null;
    let selection = selectConfidentImages([], vehicle);

    for (const query of queries) {
      const results = await Promise.all(PROVIDERS.map((p) => p.search(query, 10)));
      const combined = dedupeCandidates(results.flat());
      if (combined.length > 0) {
        const r = rankCandidates(combined, vehicle);
        const s = selectConfidentImages(r, vehicle);
        candidates = combined;
        ranked = r;
        selection = s;
        usedQuery = query;
        if (s.urls.length > 0) break;
      }
      // Rung'lar arasında da küçük bir bekleme - istek yoğunluğunu daha da azaltır.
      await new Promise((r) => setTimeout(r, 500));
    }

    let fromInfoboxFallback = false;
    if (selection.urls.length === 0) {
      const infobox = await fetchWikipediaInfobox(vehicle.make, vehicle.model);
      if (infobox.length > 0) {
        const r = rankCandidates(infobox, vehicle);
        const s = selectConfidentImages(r, vehicle);
        candidates = infobox;
        ranked = r;
        selection = s;
        fromInfoboxFallback = true;
      }
    }

    const topRanked = ranked.find((c) => c.url === selection.best.imageUrl) ?? null;

    return {
      id: tc.id,
      category: tc.category,
      rawInput: tc,
      normalized: {
        make: vehicle.make,
        model: vehicle.model,
        rawModel: vehicle.rawModel,
        variant: vehicle.variant,
        trim: vehicle.trim,
        year: vehicle.year,
        bodyType: vehicle.bodyType,
        color: vehicle.color,
      },
      resolvedGeneration: vehicle.generation,
      resolvedFacelift: vehicle.facelift,
      generationSource: vehicle.generationSource,
      generationYearRange: [vehicle.generationStartYear, vehicle.generationEndYear],
      queries,
      usedQuery,
      candidateCount: candidates.length,
      rejectedCount: ranked.filter((c) => c.rejected).length,
      rejectedReasons: ranked
        .filter((c) => c.rejected)
        .map((c) => ({ title: c.title ?? c.filename ?? c.url, reason: c.rejectionReason })),
      selected: {
        url: selection.best.imageUrl,
        title: topRanked?.title ?? topRanked?.filename ?? null,
        source: topRanked?.source ?? null,
        score: topRanked?.score ?? null,
        confidence: selection.best.confidence,
        matched: topRanked?.matched ?? null,
      },
      fromInfoboxFallback,
      error: null,
    };
  } catch (err) {
    return {
      id: tc.id,
      category: tc.category,
      rawInput: tc,
      normalized: {
        make: tc.brand,
        model: tc.model,
        rawModel: tc.model,
        variant: null,
        trim: null,
        year: tc.year ?? null,
        bodyType: tc.bodyType ?? null,
        color: tc.color ?? null,
      },
      resolvedGeneration: null,
      resolvedFacelift: null,
      generationSource: "unknown",
      generationYearRange: [null, null],
      queries: [],
      usedQuery: null,
      candidateCount: 0,
      rejectedCount: 0,
      rejectedReasons: [],
      selected: { url: null, title: null, source: null, score: null, confidence: 0, matched: null },
      fromInfoboxFallback: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function runWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const current = idx++;
      results[current] = await fn(items[current]);
      process.stdout.write(`.`);
      await new Promise((r) => setTimeout(r, DELAY_BETWEEN_CASES_MS));
    }
  }
  await Promise.all(Array.from({ length: limit }, worker));
  return results;
}

async function main() {
  console.log(`${CASES.length} vaka calistiriliyor (concurrency=${CONCURRENCY})...`);
  const results = await runWithConcurrency(CASES, CONCURRENCY, auditOne);
  console.log("\ntamamlandi.");

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonPath = `scripts/audit-results/audit-${timestamp}.json`;
  const latestJsonPath = `scripts/audit-results/audit-latest.json`;
  writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  writeFileSync(latestJsonPath, JSON.stringify(results, null, 2));

  const csvHeader = [
    "id", "category", "brand", "rawModel", "year", "canonicalModel", "variant", "trim",
    "resolvedGeneration", "facelift", "generationSource", "queries", "usedQuery",
    "candidateCount", "rejectedCount", "selectedUrl", "selectedTitle", "score", "confidence", "error",
  ].join(",");
  const csvRows = results.map((r) =>
    [
      r.id,
      r.category,
      r.rawInput.brand,
      JSON.stringify(r.rawInput.model),
      r.rawInput.year ?? "",
      JSON.stringify(r.normalized.model),
      JSON.stringify(r.normalized.variant ?? ""),
      JSON.stringify(r.normalized.trim ?? ""),
      r.resolvedGeneration ?? "",
      r.resolvedFacelift ?? "",
      r.generationSource,
      r.queries.length,
      JSON.stringify(r.usedQuery ?? ""),
      r.candidateCount,
      r.rejectedCount,
      JSON.stringify(r.selected.url ?? ""),
      JSON.stringify(r.selected.title ?? ""),
      r.selected.score ?? "",
      r.selected.confidence.toFixed(3),
      JSON.stringify(r.error ?? ""),
    ].join(",")
  );
  writeFileSync(`scripts/audit-results/audit-${timestamp}.csv`, [csvHeader, ...csvRows].join("\n"));
  writeFileSync(`scripts/audit-results/audit-latest.csv`, [csvHeader, ...csvRows].join("\n"));

  console.log(`\nSonuclar yazildi: ${jsonPath}`);
}

main();
