/**
 * Uçtan uca (orkestrasyon) testler - plan Madde 32 Case 1 (canlı BMW
 * regresyonu, gerçek Wikipedia/Commons ağı kullanır) ve Case 8 (kaynak
 * hatası - fetch mock'lanır, price prediction'ı etkilememesi gerektiği
 * doğrulanır). Case 1 ağa bağımlı olduğu için makul bir timeout ile
 * çalıştırılır.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { getVehicleImages } from "./vehicle-image.service";

test(
  "Case 1 — BMW regresyonu: 2023 BMW 3 Serisi 320i Executive M Sport Beyaz -> G20 fotoğrafı (E46 DEĞİL)",
  { timeout: 20_000 },
  async () => {
    const urls = await getVehicleImages({
      brand: "BMW",
      model: "3 Serisi",
      year: 2023,
      trim: "320i Executive M Sport",
      bodyType: "Sedan",
      color: "Beyaz",
    });

    assert.ok(urls.length > 0, "en az bir güvenilir aday dönmeli");
    assert.ok(
      urls.some((u) => u.toLowerCase().includes("g20")),
      `dönen URL'lerin en az biri G20 içermeli, aldık: ${JSON.stringify(urls)}`
    );
    assert.ok(
      !urls.some((u) => u.toLowerCase().includes("e46")),
      "hiçbir URL E46 (eski nesil) içermemeli - bu regresyonun asıl amacı"
    );
  }
);

test("Case 8 — kaynak hatası: fetch her zaman reddedilirse getVehicleImages fırlatmaz, [] döner", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => Promise.reject(new Error("network down"))) as typeof fetch;
  try {
    // Case 1'in gerçek ağ çağrıları process-içi cache'i (cache.ts) bu sorgu
    // için zaten doldurduğundan aynı aracı kullanmak fetch mock'unu hiç
    // devreye sokmaz (cache'ten döner) - benzersiz bir marka/model/yıl ile
    // cache'in kesinlikle boş olduğu, fetch'in gerçekten çağrılacağı
    // garanti edilir.
    const urls = await getVehicleImages({
      brand: "ZzTestBrandNetworkFailure",
      model: "ZzTestModelNetworkFailure",
      year: 1999,
      trim: null,
      bodyType: null,
      color: null,
    });
    assert.deepEqual(urls, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("boş marka/model erken çıkar, ağa hiç istek atmadan [] döner", async () => {
  const urls = await getVehicleImages({ brand: "", model: "" });
  assert.deepEqual(urls, []);
});
