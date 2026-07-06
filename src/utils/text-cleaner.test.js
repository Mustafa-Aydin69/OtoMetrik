const { test } = require('node:test');
const assert = require('node:assert/strict');
const { normalizeWhitespace, parseNumber, parsePriceTL, parseKilometre } = require('./text-cleaner');

test('normalizeWhitespace trims and collapses spaces', () => {
  assert.equal(normalizeWhitespace('  a   b  '), 'a b');
});

test('parseNumber strips non-digit characters', () => {
  assert.equal(parseNumber('750.000 TL'), 750000);
  assert.equal(parseNumber('68.000 km'), 68000);
});

test('parseNumber returns null for empty or non-numeric input', () => {
  assert.equal(parseNumber(''), null);
  assert.equal(parseNumber('Belirtilmemiş'), null);
});

test('parsePriceTL parses Turkish price format', () => {
  assert.equal(parsePriceTL('1.650.000 TL'), 1650000);
});

test('parseKilometre parses Turkish km format', () => {
  assert.equal(parseKilometre('45.000 km'), 45000);
});
