// Minivan & Panelvan kategorisi icin tek gecislik deneme (https://www.arabam.com/ikinci-el/minivan-panelvan).
// Site sayfalamayi 50. sayfada kesiyor (sayfa basina 20 benzersiz ilan) - yani bu kategoride
// gorunen 34.719 ilanin sadece ~1.000'ine tek gecişte erisilebiliyor, geri kalani filtreli
// alt-dilim taramasi gerektirir (bu script'in kapsami disinda). Ev IP'siyle dogrudan baglanti,
// Kafka'ya dokunmadan CsvWriter ile data/output/arabam_test_val.csv'ye ekler.
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

const CATEGORY_PATH = '/ikinci-el/minivan-panelvan';
const CATEGORY_KEY = 'minivan_panelvan';

async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);
  const csvWriter = new CsvWriter();

  try {
    const links = await collectListingLinks(page, CATEGORY_PATH);
    console.log(`${links.length} ilan linki bulundu.`);

    let processed = 0;
    let written = 0;
    for (const link of links) {
      processed += 1;

      const ilanId = link.split('/').filter(Boolean).pop();
      if (csvWriter.seenIds.has(ilanId)) {
        console.log(`[${processed}/${links.length}] ${ilanId} zaten vardi (atlandi, cekilmedi)`);
        continue;
      }

      try {
        const listing = await withRetry(() => scrapeListingDetail(page, link, CATEGORY_KEY));
        const isNew = await csvWriter.writeListing(listing);
        if (isNew) written += 1;
        console.log(`[${processed}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`);
      } catch (err) {
        console.error(`Ilan cekilemedi (${link}):`, err.message);
      } finally {
        await politeDelay();
      }
    }

    console.log(`Tamamlandi: ${written} yeni kayit yazildi (${processed} ilan islendi).`);
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Test kazima basarisiz:', err);
  process.exit(1);
});
