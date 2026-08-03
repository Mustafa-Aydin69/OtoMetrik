/**
 * Faz 13: kategori label->canonical çevirisinin regresyon testleri.
 * Çalıştırma (WebSite/ çalışma dizini olarak): npm test
 */
import assert from "node:assert/strict";
import test from "node:test";

import { toCanonicalPayload, validatePrediction, type PredictionInput } from "./validation";
import { MILEAGE_MAX, YEAR_MAX, YEAR_MIN } from "./domain-bounds.generated";

const BASE: PredictionInput = {
  brand: "Ford",
  model: "Focus",
  year: 2018,
  mileage: 85000,
  fuelType: "Benzin",
  transmission: "Otomatik",
  bodyType: "Sedan",
  color: "Beyaz",
  engineDisplacement: 1600,
  enginePower: 125,
  trim: "Titanium",
  replacedPartsCount: 1,
  paintedPartsCount: 2,
  heavyDamage: false,
};

test("Manuel etiketi -> canonical Düz", () => {
  const result = toCanonicalPayload({ ...BASE, transmission: "Manuel" });
  assert.equal(result.transmission, "Düz");
});

test("Mercedes-Benz etiketi -> canonical Mercedes - Benz", () => {
  const result = toCanonicalPayload({ ...BASE, brand: "Mercedes-Benz" });
  assert.equal(result.brand, "Mercedes - Benz");
});

test("LPG etiketi -> canonical LPG & Benzin", () => {
  const result = toCanonicalPayload({ ...BASE, fuelType: "LPG" });
  assert.equal(result.fuelType, "LPG & Benzin");
});

test("Gümüş etiketi -> canonical Gri (Gümüş)", () => {
  const result = toCanonicalPayload({ ...BASE, color: "Gümüş" });
  assert.equal(result.color, "Gri (Gümüş)");
});

test("Hatchback (3 Kapı) etiketi -> canonical Hatchback/3", () => {
  const result = toCanonicalPayload({ ...BASE, bodyType: "Hatchback (3 Kapı)" });
  assert.equal(result.bodyType, "Hatchback/3");
});

test("zaten kanonik bir değer değişmeden geçer (Otomatik -> Otomatik)", () => {
  const result = toCanonicalPayload({ ...BASE, transmission: "Otomatik" });
  assert.equal(result.transmission, "Otomatik");
});

test("kategorik olmayan alanlar değişmeden kalır", () => {
  const result = toCanonicalPayload(BASE);
  assert.equal(result.model, BASE.model);
  assert.equal(result.mileage, BASE.mileage);
  assert.equal(result.enginePower, BASE.enginePower);
});

test("yıl sınırları domain-bounds.generated.ts'ten geliyor (backend ile aynı)", () => {
  assert.equal(validatePrediction({ ...BASE, year: YEAR_MIN - 1 }).year !== undefined, true);
  assert.equal(validatePrediction({ ...BASE, year: YEAR_MAX }).year, undefined);
  assert.equal(validatePrediction({ ...BASE, year: YEAR_MAX + 1 }).year !== undefined, true);
});

test("kilometre üst sınırı artık 1.000.000 (eski 2.000.000 değil)", () => {
  assert.equal(validatePrediction({ ...BASE, mileage: MILEAGE_MAX }).mileage, undefined);
  assert.equal(validatePrediction({ ...BASE, mileage: MILEAGE_MAX + 1 }).mileage !== undefined, true);
});

test("değişen+boyalı toplamı 13'ü aşınca reddedilir", () => {
  const result = validatePrediction({ ...BASE, replacedPartsCount: 7, paintedPartsCount: 7 });
  assert.equal(result.paintedPartsCount !== undefined, true);
});
