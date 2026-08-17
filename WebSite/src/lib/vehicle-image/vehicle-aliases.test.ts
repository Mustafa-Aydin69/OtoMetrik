import { test } from "node:test";
import assert from "node:assert/strict";
import { canonicalizeModelText } from "./vehicle-aliases";

test("BMW: generic 'Serisi' -> 'Series' çevirisi (regresyon - bkz. plan Madde 1/38)", () => {
  assert.equal(canonicalizeModelText("BMW", "3 Serisi"), "3 Series");
  assert.equal(canonicalizeModelText("BMW", "5 Serisi"), "5 Series");
});

test("BMW: 'Serisi' içermeyen model adı değişmeden kalır", () => {
  assert.equal(canonicalizeModelText("BMW", "X5"), "X5");
});

test("Mercedes-Benz: 'Serisi' SİLİNİR (translate değil strip) - Wikipedia'nın kendi redirect'i bare kodu tanıyor", () => {
  assert.equal(canonicalizeModelText("Mercedes-Benz", "V Serisi"), "V");
});

test("Mercedes-Benz: zaten bare model kodu (Serisi yok) değişmeden kalır (Case 6 - canonical model 'C')", () => {
  assert.equal(canonicalizeModelText("Mercedes-Benz", "C"), "C");
});

test("Audi: 'Serisi' SİLİNİR (Mercedes ile aynı strateji, canlı Wikipedia testiyle doğrulandı)", () => {
  assert.equal(canonicalizeModelText("Audi", "100 Serisi"), "100");
});

test("Ford: default (translate) stratejisi - Wikipedia kendi redirect'iyle tireli forma yönlendiriyor", () => {
  assert.equal(canonicalizeModelText("Ford", "E Serisi"), "E Series");
});

test("Sınıfı/Sınıf -> Class çevirisi (generic, marka default stratejisiyle)", () => {
  assert.equal(canonicalizeModelText("SomeBrand", "X Sınıfı"), "X Class");
});
