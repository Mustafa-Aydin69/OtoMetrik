const { test } = require('node:test');
const assert = require('node:assert/strict');
const { parseListing, splitModelDetail } = require('./json-parser');

test('splitModelDetail separates engine code from trim', () => {
  assert.deepEqual(splitModelDetail('1.5 TSI R-Line'), { motorHacmi: '1.5 TSI', paket: 'R-Line' });
});

test('splitModelDetail handles displacement without a known engine code', () => {
  assert.deepEqual(splitModelDetail('1.6 XT'), { motorHacmi: '1.6', paket: 'XT' });
});

test('splitModelDetail handles empty input', () => {
  assert.deepEqual(splitModelDetail(''), { motorHacmi: null, paket: null });
});

test('parseListing maps a raw listing to the 17-field schema', () => {
  const raw = {
    ilanNo: 1105432901,
    kategori: 'Otomobil',
    marka: 'Volkswagen',
    seri: 'Golf',
    model: '1.5 TSI R-Line',
    kasaTipi: 'Hatchback',
    renk: 'Beyaz',
    motorGucu: 150,
    yil: 2021,
    kilometre: '45.000 km',
    yakitTipi: 'Benzin',
    vitesTipi: 'Yarı Otomatik',
    degisenSayisi: 1,
    boyaliSayisi: 2,
    fiyat: '1.650.000 TL',
  };

  const result = parseListing(raw);

  assert.equal(result.ilan_id, '1105432901');
  assert.equal(result.arac_turu, 'Otomobil');
  assert.equal(result.marka, 'Volkswagen');
  assert.equal(result.model, 'Golf');
  assert.equal(result.paket, 'R-Line');
  assert.equal(result.kasa_turu, 'Hatchback');
  assert.equal(result.renk, 'Beyaz');
  assert.equal(result.motor_hacmi, '1.5 TSI');
  assert.equal(result.motor_gucu, '150');
  assert.equal(result.yil, 2021);
  assert.equal(result.kilometre, 45000);
  assert.equal(result.yakit_turu, 'Benzin');
  assert.equal(result.vites, 'Yarı Otomatik');
  assert.equal(result.degisen_sayisi, 1);
  assert.equal(result.boyali_sayisi, 2);
  assert.equal(result.fiyat, 1650000);
  assert.ok(result.scraped_at);
});
