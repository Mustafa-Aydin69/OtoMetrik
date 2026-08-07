/**
 * Faz 13: kategori label->canonical çevirisinin regresyon testleri.
 * Çalıştırma (WebSite/ çalışma dizini olarak): npm test
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalToLabel,
  getBodyTypesForModel,
  getEnginesForModel,
  getHpOptions,
  getModelsForBrand,
  getPaketOptions,
  toCanonicalPayload,
  validatePrediction,
  type PredictionInput,
} from "./validation";
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

test("getPaketOptions: bilinen marka+model+motor için egitimdeki paketleri döner", () => {
  // Faz 25: paket artik marka+model DEGIL marka+model+motor bazinda -
  // Ford Focus 1.6 Dizel'de "TDCi Trend X" en sik gorulen paket.
  const suggestions = getPaketOptions("Ford", "Focus", 1600, "Dizel");
  assert.ok(suggestions.length > 0);
  assert.ok(suggestions.includes("TDCi Trend X"));
});

test("getPaketOptions: website etiketini (Mercedes-Benz) kanonik markaya (Mercedes - Benz) çevirir", () => {
  // Mercedes'te egitim verisindeki "model" alani "C" (BMW'deki "3 Serisi"
  // formatinin aksine "Serisi" eki YOK - markaya gore farkli scrape kalibi).
  const suggestions = getPaketOptions("Mercedes-Benz", "C", 1500, "Benzin");
  assert.ok(suggestions.length > 0);
});

test("getPaketOptions: bilinmeyen marka+model için boş dizi döner", () => {
  const suggestions = getPaketOptions("Ford", "Bilinmeyen Model XYZ", 1600, "Benzin");
  assert.deepEqual(suggestions, []);
});

test("getEnginesForModel: Mazda 3 için motor kombinasyonlarını hacme göre artan döner", () => {
  const engines = getEnginesForModel("Mazda", "3");
  assert.ok(engines.length > 0);
  assert.ok(engines.some((e) => e.hacmiBucket === 1600 && e.yakitTuru === "Benzin"));
  for (let i = 1; i < engines.length; i++) {
    assert.ok(engines[i - 1].hacmiBucket <= engines[i].hacmiBucket);
  }
});

test("getEnginesForModel: exactCc kovadaki en sık görülen gerçek cc değeri (kovanın kendisi değil)", () => {
  const engines = getEnginesForModel("Mazda", "3");
  const bucket1600Benzin = engines.find((e) => e.hacmiBucket === 1600 && e.yakitTuru === "Benzin");
  assert.ok(bucket1600Benzin);
  assert.equal(bucket1600Benzin!.exactCc, 1598);
});

test("getEnginesForModel: bilinmeyen marka+model için boş dizi döner", () => {
  assert.deepEqual(getEnginesForModel("Ford", "Bilinmeyen Model XYZ"), []);
});

test("getHpOptions: birden fazla geçerli motor gücü değeri olan gerçek bir kombinasyon", () => {
  // Bu Faz'da olculdu: Mazda 3 1.6 Benzin'de 3 farkli HP degeri var -
  // PredictionForm bu durumda otomatik doldurmak yerine secilebilir dropdown gosterir.
  const hp = getHpOptions("Mazda", "3", 1600, "Benzin");
  assert.deepEqual(hp, [105, 109, 115]);
});

test("getHpOptions: bilinmeyen kombinasyon için boş dizi döner", () => {
  assert.deepEqual(getHpOptions("Ford", "Bilinmeyen Model XYZ", 1600, "Benzin"), []);
});

test("canonicalToLabel: labelToCanonical'ın tersi (LPG & Benzin -> LPG)", () => {
  assert.equal(canonicalToLabel("fuelType", "LPG & Benzin"), "LPG");
  assert.equal(canonicalToLabel("fuelType", "Dizel"), "Dizel");
});

test("canonicalToLabel: eşleşme yoksa değeri olduğu gibi döner", () => {
  assert.equal(canonicalToLabel("fuelType", "Bilinmeyen Kanonik Değer"), "Bilinmeyen Kanonik Değer");
});

test("getBodyTypesForModel: Citroën Berlingo için sadece gerçekten görülen kasa tiplerini döner", () => {
  // Berlingo satırlarının %87'si "Camlı Van" - "Sedan"/"MPV" gibi bu araca
  // hiç uymayan sabit-liste seçenekleri artık dönmüyor (bkz. kullanıcı raporu).
  const types = getBodyTypesForModel("Citroën", "Berlingo");
  assert.ok(types.length > 0);
  assert.equal(types[0], "Camlı Van");
  assert.ok(!types.includes("Sedan"));
  assert.ok(!types.includes("MPV"));
});

test("getBodyTypesForModel: bilinmeyen marka+model için boş dizi döner", () => {
  assert.deepEqual(getBodyTypesForModel("Ford", "Bilinmeyen Model XYZ"), []);
});

test("getHpOptions: Citroën Berlingo 1.5 Dizel'de yakın-deger olcum gurultusu 5 HP'ye yuvarlanip birlestirilir", () => {
  // Kullanici raporu: 96/100/102/110/130/132 -> 6 secenek gorunuyordu.
  // 100~102 ve 130~132 ayni gercek versiyonun olcum gurultusu (bkz.
  // generate_vehicle_options.py Faz 26 notu) - yuvarlama sonrasi 4 kalmali,
  // baskin gercek degerler (102 ve 132, en sik gorulenler) korunmali.
  const hp = getHpOptions("Citroën", "Berlingo", 1500, "Dizel");
  assert.equal(hp.length, 4);
  assert.ok(hp.includes(102));
  assert.ok(hp.includes(132));
});

test("getModelsForBrand: BMW için modelleri döner ve 3 Serisi içerir", () => {
  const models = getModelsForBrand("BMW");
  assert.ok(models.length > 0);
  assert.ok(models.includes("3 Serisi"));
});

test("getModelsForBrand: website etiketini (Mercedes-Benz) kanonik markaya çevirir", () => {
  // Mercedes'te egitim verisindeki model degeri "C" (getPaketOptions
  // testindeki notla tutarli) - VehicleSelector'ın model listesinde de aynı
  // ham değer görünür.
  const models = getModelsForBrand("Mercedes-Benz");
  assert.ok(models.length > 0);
  assert.ok(models.includes("C"));
});

test("getModelsForBrand: genişletilmiş marka listesindeki bir marka (Mazda) için de model döner", () => {
  // 19 markalık eski MARKA_OPTIONS'ta Mazda yoktu; Faz 24 genişletmesiyle
  // eklendi - bu test genişletmenin gerçekten uçtan uca bağlandığının kanıtı.
  const models = getModelsForBrand("Mazda");
  assert.ok(models.length > 0);
});

test("getModelsForBrand: bilinmeyen marka için boş dizi döner", () => {
  assert.deepEqual(getModelsForBrand("Bilinmeyen Marka XYZ"), []);
});

test("getModelsForBrand: dönen liste tr-collator sırasına göre sıralı", () => {
  const models = getModelsForBrand("BMW");
  const collator = new Intl.Collator("tr", { numeric: true, sensitivity: "base" });
  const sorted = [...models].sort(collator.compare);
  assert.deepEqual(models, sorted);
});
