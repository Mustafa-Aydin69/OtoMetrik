// Minivan & Panelvan kategorisi kategori sayfasi tek basina sadece ~2.500 benzersiz ilana
// (50 sayfa x take=50) kadar erisim veriyor - ama site "34.339 sonuc" diyor. Aradaki fark,
// sitenin sayfalamayi 50. sayfada kesmesinden kaynaklaniyor (bkz. scrape-minivan-panelvan-test.js
// yorumu). Her marka kendi filtrelenmis sayfasinda AYRI bir 2.500'luk sayfalama hakkina sahip
// oldugu icin, kategoriyi markaya gore dilimleyerek cok daha fazla benzersiz ilana ulasabiliyoruz
// (kullanicinin linkler.txt'de verdigi marka basi ilan sayilarina gore tahmini ~19.000).
// Zaten CSV'de olan ilanlar agdan cekilmeden atlaniyor (ayni resume garantisi).
//
// Bazi markalar (Fiat 9.970, Ford 8.065, VW 4.652, Peugeot 2.915, Renault 2.531) tek basina
// da 2.500 sinirini asiyor (Renault'da canli dogrulandi: 2.531 ilanin sadece 2.500'u islendi).
// Bu markalar icin markayi ayrica YIL araligina gore diliyoruz - arabam.com "minYear"/"maxYear"
// query paramlarini destekliyor (kullanicinin verdigi ornek: .../peugeot?minYear=2023). Yil,
// km'nin aksine ilanin degismez ozelligi oldugu icin dilim sinirlari tekrar calistirmalarda
// kararli kalir. Dilimler CsvWriter.seenIds ile ilan_id bazinda tekillestigi icin cakismalari
// sorun degil - sadece "bosluk" (hicbir dilime dusmeyen ilan) birakmamak yeterli.
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

// linkler.txt'de kullanicinin verdigi marka sayfalari, gerektiginde yil dilimiyle birlikte.
// Her eleman { brand, minYear?, maxYear? } - minYear/maxYear yoksa markanin tamami tek dilimde
// cekilir (2.500 altinda kalan markalar icin yeterli).
const SLICES = [
  //{ brand: 'citroen' },
  //{ brand: 'dacia' },
  //{ brand: 'hyundai' },
  //{ brand: 'mercedes-benz' },
  //{ brand: 'mitsubishi' },
  //{ brand: 'opel' },
  //{ brand: 'peugeot', minYear: 2023 },
  //{ brand: 'peugeot', maxYear: 2023 },
  //{ brand: 'renault' },
  //{ brand: 'toyota' },
  //{ brand: 'fiat', maxYear: 2010 },
  //{ brand: 'fiat', minYear: 2010, maxYear: 2013 },
  //{ brand: 'fiat', minYear: 2013, maxYear: 2016 },
  //{ brand: 'fiat', minYear: 2016, maxYear: 2020 },
  //{ brand: 'fiat', minYear: 2020, maxYear: 2023 },
  //{ brand: 'fiat', minYear: 2023 },
  //{ brand: 'ford', maxYear: 2013 },
  //{ brand: 'ford', minYear: 2013, maxYear: 2018 },
  //{ brand: 'ford', minYear: 2018, maxYear: 2022 },
  //{ brand: 'ford', minYear: 2022 },
  { brand: 'volkswagen', maxYear: 2014 },
  { brand: 'volkswagen', minYear: 2014 },
];

function sliceLabel(slice) {
  if (slice.minYear && slice.maxYear) return `${slice.brand} (${slice.minYear}-${slice.maxYear})`;
  if (slice.minYear) return `${slice.brand} (${slice.minYear}+)`;
  if (slice.maxYear) return `${slice.brand} (-${slice.maxYear})`;
  return slice.brand;
}

async function scrapeBrand(page, csvWriter, slice) {
  const label = sliceLabel(slice);
  let categoryPath = `/ikinci-el/minivan-panelvan/${slice.brand}?take=50`;
  if (slice.minYear) categoryPath += `&minYear=${slice.minYear}`;
  if (slice.maxYear) categoryPath += `&maxYear=${slice.maxYear}`;

  const links = await collectListingLinks(page, categoryPath);
  console.log(`\n=== ${label}: ${links.length} ilan linki bulundu ===`);

  let processed = 0;
  let written = 0;
  for (const link of links) {
    processed += 1;

    const ilanId = link.split('/').filter(Boolean).pop();
    if (csvWriter.seenIds.has(ilanId)) {
      console.log(`[${label} ${processed}/${links.length}] ${ilanId} zaten vardi (atlandi, cekilmedi)`);
      continue;
    }

    try {
      const listing = await withRetry(() => scrapeListingDetail(page, link, CATEGORY_KEY));
      const isNew = await csvWriter.writeListing(listing);
      if (isNew) written += 1;
      console.log(`[${label} ${processed}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`);
    } catch (err) {
      console.error(`Ilan cekilemedi (${link}):`, err.message);
    } finally {
      await politeDelay();
    }
  }

  console.log(`--- ${label} tamamlandi: ${written} yeni kayit (${processed} ilan islendi) ---`);
  return { label, processed, written };
}

async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);
  const csvWriter = new CsvWriter();

  const results = [];
  try {
    for (const slice of SLICES) {
      const result = await scrapeBrand(page, csvWriter, slice);
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
    console.log(`${r.label}: ${r.written} yeni / ${r.processed} islendi`);
  }
  console.log(`TOPLAM: ${totalWritten} yeni kayit yazildi (${totalProcessed} ilan islendi).`);
}

run().catch((err) => {
  console.error('Marka bazli kazima basarisiz:', err);
  process.exit(1);
});
