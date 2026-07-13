// Elektrikli araclar kategorisi (891 ilan, https://www.arabam.com/ikinci-el/otomobil-elektrik)
// icin kucuk olcekli deneme. Eski sisteme donuldu: proxy denemeleri (Webshare, Bright Data ISP,
// Bright Data Scraping Browser) iptal edildi, dogrudan ev IP'siyle baglaniliyor (dunku 1903
// kayitlik basarili kazimayla ayni yontem). Kafka'ya dokunmadan dogrudan CsvWriter ile
// data/output/arabam_test_val.csv'ye ekler.
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

const CATEGORY_PATH = '/ikinci-el/otomobil-elektrik';
const CATEGORY_KEY = 'elektrikli';

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

      // Onceki (kesintiye ugramis) calismadan zaten CSV'de olan ilanlari agdan hic cekmeden
      // atla - "kaldigi yerden devam" hizli olsun diye, ayni ilani tekrar indirmeye gerek yok.
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
