/**
 * Audit yardımcı script'i (üretim koduna DOKUNMAZ) - vehicle-options.generated.ts'teki
 * ENGINES_BY_MODEL'in count alanlarını toplayarak marka/model dağılımını hesaplar.
 * Gerçek CSV'yi (46MB) parse etmeye gerek yok - count zaten eğitim verisindeki
 * gerçek satır sayısını yansıtıyor (bkz. generate_vehicle_options.py).
 *
 * Çalıştırma: npx tsx scripts/dataset-frequency.ts
 */
import { writeFileSync } from "node:fs";
import { ENGINES_BY_MODEL, MODELS_BY_BRAND } from "../src/lib/vehicle-options.generated";

const modelCount = new Map<string, number>();
for (const [key, engines] of Object.entries(ENGINES_BY_MODEL)) {
  const parts = key.split("|");
  const brand = parts[0];
  const model = parts[1];
  const mkKey = `${brand}|${model}`;
  const sum = engines.reduce((s, e) => s + (e.count ?? 0), 0);
  modelCount.set(mkKey, (modelCount.get(mkKey) ?? 0) + sum);
}

const brandCount = new Map<string, number>();
for (const [key, count] of modelCount) {
  const brand = key.split("|")[0];
  brandCount.set(brand, (brandCount.get(brand) ?? 0) + count);
}

const sortedBrands = [...brandCount.entries()].sort((a, b) => b[1] - a[1]);
console.log("=== TOP 40 BRANDS BY DATASET COUNT ===");
for (const [brand, count] of sortedBrands.slice(0, 40)) {
  console.log(`${brand}\t${count}\t${MODELS_BY_BRAND[brand]?.length ?? 0} models`);
}

console.log("\n=== TOP 100 MAKE|MODEL BY COUNT ===");
const sortedModels = [...modelCount.entries()].sort((a, b) => b[1] - a[1]);
for (const [key, count] of sortedModels.slice(0, 100)) {
  console.log(`${key}\t${count}`);
}

console.log(`\nTotal unique make|model combos: ${modelCount.size}`);
console.log(`Total brands: ${brandCount.size}`);

writeFileSync(
  "scripts/audit-results/dataset-frequency.json",
  JSON.stringify(
    {
      brands: sortedBrands.map(([make, count]) => ({ make, count, modelCount: MODELS_BY_BRAND[make]?.length ?? 0 })),
      models: sortedModels.map(([key, count]) => {
        const [make, model] = key.split("|");
        return { make, model, count };
      }),
    },
    null,
    2
  )
);
