/**
 * Audit yardımcı script'i (üretim koduna DOKUNMAZ) - generation-rules.ts ve
 * vehicle-aliases.ts'in dataset-frequency.json'daki GERÇEK marka/model
 * dağılımına göre kapsama oranını hesaplar (bkz. audit görevi Madde 9/10).
 *
 * Çalıştırma: npx tsx scripts/coverage-analysis.ts (önce dataset-frequency.ts çalıştırılmış olmalı)
 */
import { readFileSync, writeFileSync } from "node:fs";
import { GENERATION_RULES } from "../src/lib/vehicle-image/generation-rules";
import { canonicalizeModelText } from "../src/lib/vehicle-image/vehicle-aliases";
import { canonicalToLabel } from "../src/lib/validation";

// dataset-frequency.ts, vehicle-options.generated.ts'in KANONİK marka
// anahtarını kullanıyor (örn. "Mercedes - Benz", boşluklu) - ama gerçek
// pipeline'a (ve generation-rules.ts'e) giden değer form ETİKETİdir (örn.
// "Mercedes-Benz", boşluksuz, bkz. category-options.generated.ts). Bu
// farkı burada normalize etmezsek Mercedes-Benz kuralları YANLIŞ ŞEKİLDE
// "kapsanmıyor" görünür (bir script/etiket uyumsuzluğu, gerçek bir kod
// boşluğu DEĞİL).
function toLabel(canonicalMake: string): string {
  return canonicalToLabel("brand", canonicalMake);
}

interface FreqEntry {
  make: string;
  model: string;
  count: number;
}

const data = JSON.parse(readFileSync("scripts/audit-results/dataset-frequency.json", "utf-8")) as {
  models: FreqEntry[];
};

const ruleCoverage = new Set(GENERATION_RULES.map((r) => `${r.make}|${r.model}`));

console.log("=== GENERATION RULE COVERAGE (top 40 make|model by dataset count) ===");
console.log("Make\tModel\tCount\tCanonicalModel\tRuleExists");
const rows: Array<Record<string, unknown>> = [];
let totalCount = 0;
for (const { count } of data.models) {
  totalCount += count;
}
for (const { make, model, count } of data.models.slice(0, 40)) {
  const brandLabel = toLabel(make);
  const canonicalModel = canonicalizeModelText(brandLabel, model);
  const hasRule = ruleCoverage.has(`${brandLabel}|${canonicalModel}`);
  console.log(`${brandLabel}\t${model}\t${count}\t${canonicalModel}\t${hasRule ? "YES" : "NO"}`);
  rows.push({ make: brandLabel, model, count, canonicalModel, hasLocalRule: hasRule });
}

const top40Total = data.models.slice(0, 40).reduce((s, m) => s + m.count, 0);
const top40Covered = rows.filter((r) => r.hasLocalRule).reduce((s, r) => s + (r.count as number), 0);
console.log(`\nTop-40 make|model toplam kayıt: ${top40Total}`);
console.log(`Bunlardan local generation rule ile kapsanan: ${top40Covered} (%${((top40Covered / top40Total) * 100).toFixed(1)})`);
console.log(`Tüm dataset toplam kayıt: ${totalCount}`);

console.log("\n=== ALIAS TRANSFORMATION AUDIT (top 60 - hangi modeller hala 'Serisi/Sınıf' iceriyor ya da degismiyor) ===");
const aliasRows: Array<Record<string, unknown>> = [];
for (const { make, model, count } of data.models.slice(0, 60)) {
  const brandLabel = toLabel(make);
  const canonical = canonicalizeModelText(brandLabel, model);
  const stillHasTurkish = /serisi|sınıf/i.test(canonical);
  aliasRows.push({ make: brandLabel, model, count, canonical, stillHasTurkish, changed: canonical !== model });
  if (stillHasTurkish) {
    console.log(`UYARI: ${brandLabel} | "${model}" -> "${canonical}" hala Turkce icin barindiriyor`);
  }
}

writeFileSync(
  "scripts/audit-results/coverage-analysis.json",
  JSON.stringify({ generationRuleCoverageTop40: rows, top40Total, top40Covered, aliasAuditTop60: aliasRows }, null, 2)
);
console.log("\nYazildi: scripts/audit-results/coverage-analysis.json");
