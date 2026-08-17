import { test } from "node:test";
import assert from "node:assert/strict";
import { generateQueries, MAX_QUERIES } from "./query-generator";
import type { VehicleIdentity } from "./types";

function bmwG20(): VehicleIdentity {
  return {
    make: "BMW",
    model: "3 Series",
    rawModel: "3 Serisi",
    variant: "320i",
    trim: "Executive M Sport",
    year: 2023,
    bodyType: "Sedan",
    color: "White",
    generation: "G20",
    generationOrdinalLabel: "Seventh generation",
    facelift: "LCI",
    generationStartYear: 2018,
    generationEndYear: null,
    generationSource: "local",
  };
}

test("generation biliniyorsa ladder ASLA generation'sız bir rung üretmez (regresyon - plan Madde 17)", () => {
  const queries = generateQueries(bmwG20());
  assert.ok(queries.length > 0);
  for (const q of queries) {
    assert.ok(q.includes("G20"), `"${q}" generation (G20) içermiyor`);
  }
});

test("en spesifik rung tüm alanları içerir, sırayla daha genele düşer", () => {
  const queries = generateQueries(bmwG20());
  assert.equal(queries[0], "2023 BMW 320i G20 LCI Executive M Sport Sedan White");
  assert.ok(queries[queries.length - 1].includes("BMW"));
  assert.ok(queries[queries.length - 1].includes("G20"));
});

test("MAX_QUERIES sınırına uyulur ve tekrar eden sorgular tekilleştirilir", () => {
  const queries = generateQueries(bmwG20());
  assert.ok(queries.length <= MAX_QUERIES);
  assert.equal(new Set(queries).size, queries.length);
});

test("generation bilinmiyorsa ladder model/make seviyesine kadar iner", () => {
  const vehicle: VehicleIdentity = {
    make: "Toyota",
    model: "Corolla",
    rawModel: "Corolla",
    variant: null,
    trim: null,
    year: 2020,
    bodyType: null,
    color: null,
    generation: null,
    generationOrdinalLabel: null,
    facelift: null,
    generationStartYear: null,
    generationEndYear: null,
    generationSource: "unknown",
  };
  const queries = generateQueries(vehicle);
  assert.ok(queries.includes("Toyota Corolla"));
  assert.ok(queries.includes("Toyota"));
});
