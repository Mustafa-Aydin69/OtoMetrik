// Minivan & Panelvan kategorisi icin genisletilmis gecis (https://www.arabam.com/ikinci-el/minivan-panelvan).
// Site sayfalamayi 50. sayfada kesiyor. "?take=50" ile sayfa basina 50 ilan gelir (varsayilan 20
// yerine) ve site "sonraki sayfa" linklerinde bunu korur - boylece 50x50=~2.500 benzersiz ilana
// kadar erisilebiliyor (once take=20 ile ~1.000'ini zaten cektik). Zaten CSV'de olanlar agdan
// cekilmeden atlandigi icin onceki ~940 kayit tekrar indirilmez. Ev IP'siyle dogrudan baglanti,
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

const CATEGORY_PATH = '/ikinci-el/minivan-panelvan?take=50';
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
