/**
 * PREDICTION_API_URL tanımlı olmadığında kullanılan deterministik mock tahmin.
 * Aynı girdi her zaman aynı fiyatı üretir (seed = girdinin FNV-1a hash'i);
 * gerçek modele geçince route.ts'teki dallanma bu dosyayı devre dışı bırakır.
 */
import type { PredictionInput } from "./validation";

function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

const BRAND_BASE: Record<string, number> = {
  "Mercedes-Benz": 1_900_000,
  BMW: 1_800_000,
  Audi: 1_750_000,
  Volvo: 1_450_000,
  Volkswagen: 1_250_000,
  Toyota: 1_200_000,
  Honda: 1_150_000,
  Skoda: 1_100_000,
  Ford: 1_000_000,
  Hyundai: 1_000_000,
  Nissan: 980_000,
  Kia: 980_000,
  Seat: 960_000,
  Renault: 950_000,
  Peugeot: 940_000,
  Opel: 930_000,
  "Citroën": 900_000,
  Fiat: 850_000,
  Dacia: 800_000,
};

export function mockPredict(input: PredictionInput): number {
  const currentYear = new Date().getFullYear();
  const age = Math.max(0, currentYear - input.year);

  let price = BRAND_BASE[input.brand] ?? 900_000;

  // Yaş amortismanı: yılda ~%8, tabanı %18'de kes.
  price *= Math.max(0.18, Math.pow(0.92, age));

  // Kilometre etkisi: yüksek km fiyatı düşürür, %45 tabanı var.
  price *= Math.max(0.45, 1 - input.mileage / 1_300_000);

  // Güç primi: 100 HP referans.
  price *= 1 + Math.max(-0.2, (input.enginePower - 100) / 650);

  // Yakıt / vites düzeltmeleri.
  if (input.fuelType === "Elektrik") price *= 1.15;
  else if (input.fuelType === "Hibrit") price *= 1.08;
  else if (input.fuelType === "LPG") price *= 0.93;
  if (input.transmission === "Otomatik") price *= 1.06;

  if (input.bodyType === "SUV") price *= 1.1;
  else if (input.bodyType === "Coupe" || input.bodyType === "Cabrio") price *= 1.05;

  // Hasar geçmişi.
  price *= Math.pow(0.98, input.replacedPartsCount);
  price *= Math.pow(0.99, input.paintedPartsCount);
  if (input.heavyDamage) price *= 0.62;

  // Deterministik jitter (±%4) — aynı girdi aynı sonucu verir.
  const seed = fnv1a(JSON.stringify(input));
  const jitter = 0.96 + (seed % 8001) / 100_000; // 0.96 – 1.04008
  price *= jitter;

  return Math.max(50_000, Math.round(price / 1000) * 1000);
}
