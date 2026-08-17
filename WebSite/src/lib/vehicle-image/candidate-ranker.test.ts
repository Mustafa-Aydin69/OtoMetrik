/**
 * candidate-ranker.ts + confidence.ts için sahte (network'süz) adaylarla
 * birim testler - plan Madde 32'deki Case 2/3/4/5 senaryoları.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { rankCandidates } from "./candidate-ranker";
import { selectConfidentImages } from "./confidence";
import type { ImageCandidate, VehicleIdentity } from "./types";

function bmwG20(overrides: Partial<VehicleIdentity> = {}): VehicleIdentity {
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
    ...overrides,
  };
}

function candidate(title: string, categories: string[] = []): ImageCandidate {
  return { url: `https://example.com/${encodeURIComponent(title)}.jpg`, title, filename: title, categories, source: "commons" };
}

test("Case 2 — generation conflict: E46 candidate, G20 target -> hard reject", () => {
  const vehicle = bmwG20();
  const candidates = [candidate("BMW320i E46 Lim")];
  const ranked = rankCandidates(candidates, vehicle);
  assert.equal(ranked[0].rejected, true);
  assert.match(ranked[0].rejectionReason ?? "", /generation conflict/);
});

test("Case 5 — correct color but wrong generation: hard reject (renk eşleşmesi ret'i engellemiyor)", () => {
  const vehicle = bmwG20();
  const candidates = [candidate("BMW E46 320i White Sedan")];
  const ranked = rankCandidates(candidates, vehicle);
  assert.equal(ranked[0].rejected, true);
});

test("Case 4 — wrong color but right generation: kabul edilir, sadece küçük penalty", () => {
  const vehicle = bmwG20();
  const candidates = [candidate("BMW 3 Series G20 Sedan Black", ["BMW G20"])];
  const ranked = rankCandidates(candidates, vehicle);
  assert.equal(ranked[0].rejected, false);
  assert.equal(ranked[0].matched.generation, true);
  assert.equal(ranked[0].matched.color, false);

  const selection = selectConfidentImages(ranked, vehicle);
  assert.ok(selection.urls.length > 0, "generation doğru olduğu için renk yanlış olsa da gösterilmeli");
});

test("Case 3 — unknown generation: candidate reddedilmez ama confidence, generation doğrulanmış duruma göre daha düşük olmalı", () => {
  const known = bmwG20();
  const unknown = bmwG20({
    generation: null,
    facelift: null,
    generationOrdinalLabel: null,
    generationStartYear: null,
    generationEndYear: null,
    generationSource: "unknown",
  });

  const knownCandidates = [candidate("BMW 3 Series G20 320i Executive M Sport Sedan White", ["BMW G20"])];
  const unknownCandidates = [candidate("BMW 320i Sedan")];

  const rankedKnown = rankCandidates(knownCandidates, known);
  const rankedUnknown = rankCandidates(unknownCandidates, unknown);

  assert.equal(rankedUnknown[0].rejected, false, "generation bilinmiyorsa candidate otomatik reddedilmemeli");

  const selKnown = selectConfidentImages(rankedKnown, known);
  const selUnknown = selectConfidentImages(rankedUnknown, unknown);

  assert.ok(selKnown.best.confidence > selUnknown.best.confidence, "generation doğrulanmış eşleşme daha yüksek confidence'a sahip olmalı");
});

test("Case 7 — sadece alakasız adaylar: hiçbir aday güven eşiğini geçemez, imageUrl null", () => {
  const vehicle = bmwG20();
  const candidates = [candidate("Random unrelated photo"), candidate("Some other file")];
  const ranked = rankCandidates(candidates, vehicle);
  const selection = selectConfidentImages(ranked, vehicle);
  assert.equal(selection.urls.length, 0);
  assert.equal(selection.best.imageUrl, null);
});

test("Model conflict (BMW 5 Series candidate, 3 Series target) hard reject edilir", () => {
  const vehicle = bmwG20();
  const candidates = [candidate("BMW 5 Series G20-era Sedan")];
  const ranked = rankCandidates(candidates, vehicle);
  assert.equal(ranked[0].rejected, true);
  assert.match(ranked[0].rejectionReason ?? "", /model conflict/);
});
