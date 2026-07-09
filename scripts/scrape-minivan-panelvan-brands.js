// Minivan & Panelvan kategorisi kategori sayfasi tek basina sadece ~2.500 benzersiz ilana
// (50 sayfa x take=50) kadar erisim veriyor - ama site "34.339 sonuc" diyor. Aradaki fark,
// sitenin sayfalamayi 50. sayfada kesmesinden kaynaklaniyor (bkz. scrape-minivan-panelvan-test.js
// yorumu). Her marka kendi filtrelenmis sayfasinda AYRI bir 2.500'luk sayfalama hakkina sahip
// oldugu icin, kategoriyi markaya gore dilimleyerek cok daha fazla benzersiz ilana ulasabiliyoruz
// (kullanicinin linkler.txt'de verdigi marka basi ilan sayilarina gore tahmini ~19.000).
// Zaten CSV'de olan ilanlar agdan cekilmeden atlaniyor (ayni resume garantisi).
require('dotenv').config();
const { CsvWriter } = require('../src/consumers/csv-writer');
const {
  launchBrowser,
  createPage,
  collectListingLinks,
  scrapeListingDetail,
  withRetry,
  politeDelay,
} = require('../src/producers/arabam-scraper');

const CATEGORY_KEY = 'minivan_panelvan';

// linkler.txt'de kullanicinin verdigi marka sayfalari (fiyat/marka filtresi ile).
const BRANDS = [
  //'citroen',
  //'dacia',
  //'fiat',
  //'ford',
  //'hyundai',
  'mercedes-benz',
  'mitsubishi',
  'opel',
  'peugeot',
  'renault',
  'toyota',
  'volkswagen',
];

async function scrapeBrand(page, csvWriter, brand) {
  const categoryPath = `/ikinci-el/minivan-panelvan/${brand}?take=50`;
  const links = await collectListingLinks(page, categoryPath);
  console.log(`\n=== ${brand}: ${links.length} ilan linki bulundu ===`);

  let processed = 0;
  let written = 0;
  for (const link of links) {
    processed += 1;

    const ilanId = link.split('/').filter(Boolean).pop();
    if (csvWriter.seenIds.has(ilanId)) {
      console.log(`[${brand} ${processed}/${links.length}] ${ilanId} zaten vardi (atlandi, cekilmedi)`);
      continue;
    }

    try {
      const listing = await withRetry(() => scrapeListingDetail(page, link, CATEGORY_KEY));
      const isNew = await csvWriter.writeListing(listing);
      if (isNew) written += 1;
      console.log(`[${brand} ${processed}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`);
    } catch (err) {
      console.error(`Ilan cekilemedi (${link}):`, err.message);
    } finally {
      await politeDelay();
    }
  }

  console.log(`--- ${brand} tamamlandi: ${written} yeni kayit (${processed} ilan islendi) ---`);
  return { brand, processed, written };
}

async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);
  const csvWriter = new CsvWriter();

  const results = [];
  try {
    for (const brand of BRANDS) {
      const result = await scrapeBrand(page, csvWriter, brand);
      results.push(result);
      await politeDelay();
    }
  } finally {
    await context.close();
    await browser.close();
  }

  const totalWritten = results.reduce((sum, r) => sum + r.written, 0);
  const totalProcessed = results.reduce((sum, r) => sum + r.processed, 0);
  console.log('\n=== OZET ===');
  for (const r of results) {
    console.log(`${r.brand}: ${r.written} yeni / ${r.processed} islendi`);
  }
  console.log(`TOPLAM: ${totalWritten} yeni kayit yazildi (${totalProcessed} ilan islendi).`);
}

run().catch((err) => {
  console.error('Marka bazli kazima basarisiz:', err);
  process.exit(1);
});
