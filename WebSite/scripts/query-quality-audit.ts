/**
 * Audit yardımcı script'i (üretim koduna DOKUNMAZ, AĞ ÇAĞRISI YAPMAZ) - audit
 * görevi Madde 11: generateQueries() çıktısını, gerçek nesil çözümlemesi
 * gerektirmeden statik olarak denetler. Yerel generation-rules.ts kuralı olan
 * araçlar için GERÇEK generation (Wikipedia'ya gitmeden) kullanılır; kalanlar
 * için "generation bilinmiyor" varsayımıyla (Wikipedia da rate-limit'li/kapalı
 * olsaydı ne olurdu senaryosu) test edilir - bu da ayrıca değerli bir
 * worst-case kontrolüdür.
 *
 * Çalıştırma: npx tsx scripts/query-quality-audit.ts
 */
import { writeFileSync } from "node:fs";
import { resolveLocalGeneration, nextRuleStartYear } from "../src/lib/vehicle-image/generation-rules";
import { generateQueries } from "../src/lib/vehicle-image/query-generator";
import type { VehicleIdentity } from "../src/lib/vehicle-image/types";
import { normalizeVehicleInput } from "../src/lib/vehicle-image/vehicle-parser";
import { CASES } from "./audit-cases";

interface QueryIssue {
  id: string;
  type: "duplicate" | "turkish-leftover" | "generation-dropped" | "too-long" | "too-generic-only";
  detail: string;
}

const issues: QueryIssue[] = [];
const rows: Array<Record<string, unknown>> = [];

for (const tc of CASES) {
  const base = normalizeVehicleInput(tc);
  const localRule = resolveLocalGeneration(base.make, base.model, base.year);
  const vehicle: VehicleIdentity = localRule
    ? {
        ...base,
        generation: localRule.generation,
        generationOrdinalLabel: localRule.ordinalLabel,
        facelift: localRule.facelift ?? null,
        generationStartYear: localRule.startYear,
        generationEndYear: localRule.endYear ?? (nextRuleStartYear(localRule) !== null ? nextRuleStartYear(localRule)! - 1 : null),
        generationSource: "local",
      }
    : base; // yerel kural yoksa generation=null (Wikipedia denenmedi - statik/offline analiz)

  const queries = generateQueries(vehicle);

  // 1) duplicate
  if (new Set(queries).size !== queries.length) {
    issues.push({ id: tc.id, type: "duplicate", detail: `${queries.length} sorgudan ${queries.length - new Set(queries).size} tanesi tekrar` });
  }

  // 2) Turkish leftover (Serisi/Sınıf kelimesi hala sorguda mı)
  for (const q of queries) {
    if (/serisi|sınıf/i.test(q)) {
      issues.push({ id: tc.id, type: "turkish-leftover", detail: q });
    }
  }

  // 3) generation dropped when known
  if (vehicle.generation) {
    const missing = queries.filter((q) => !q.includes(vehicle.generation!));
    if (missing.length > 0) {
      issues.push({ id: tc.id, type: "generation-dropped", detail: `${missing.length}/${queries.length} sorgu "${vehicle.generation}" içermiyor: ${missing.join(" | ")}` });
    }
  }

  // 4) too long (>80 karakter garip bir sorgu, Commons full-text search için gereksiz uzun)
  for (const q of queries) {
    if (q.length > 80) {
      issues.push({ id: tc.id, type: "too-long", detail: `${q.length} karakter: "${q}"` });
    }
  }

  // 5) yalnızca marka (en genel rung) TEK sorgu değilse - normal, ama marka-only rung'un
  //    var olup olmadığı ve kaç adet "çok genel" (<=1 kelime) sorgu üretildiği raporlanır.
  const genericCount = queries.filter((q) => q.trim().split(/\s+/).length <= 1).length;

  rows.push({
    id: tc.id,
    make: vehicle.make,
    model: vehicle.model,
    generation: vehicle.generation,
    generationSource: localRule ? "local" : "unknown(offline-test)",
    queryCount: queries.length,
    queries,
    genericCount,
  });
}

console.log(`${CASES.length} vaka statik olarak denetlendi (ağ çağrısı YOK).`);
console.log(`\nSorunlar: ${issues.length}`);
for (const issue of issues) {
  console.log(`  [${issue.type}] ${issue.id}: ${issue.detail}`);
}

writeFileSync("scripts/audit-results/query-quality-audit.json", JSON.stringify({ issues, rows }, null, 2));
console.log("\nYazildi: scripts/audit-results/query-quality-audit.json");
