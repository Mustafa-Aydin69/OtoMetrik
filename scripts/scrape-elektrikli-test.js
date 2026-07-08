// Ilk proxy rotasyon denemesi: arabam.com'un elektrikli araclar kategorisini (891 ilan, bkz.
// https://www.arabam.com/ikinci-el/otomobil-elektrik) kazir. Kafka'ya dokunmadan dogrudan
// CsvWriter ile data/output/arabam_test_val.csv'ye ekler - production akisindaki (arabam-scraper.js
// run()) proxy rotasyonu/retry/rate-limit mantigini aynen kullanir, sadece tek kategoriye ozel.
require('dotenv').config();
const { CsvWriter } = require('../src/consumers/csv-writer');
const {
  launchBrowser,
  createContext,
  createPage,
  collectListingLinks,
  scrapeListingDetail,
  withRetry,
  politeDelay,
} = require('../src/producers/arabam-scraper');
const scraperRules = require('../config/scraper-rules.json');

const CATEGORY_PATH = '/ikinci-el/otomobil-elektrik';
const CATEGORY_KEY = 'elektrikli';

async function run() {
  const { browser, context: initialContext } = await launchBrowser();
  let context = initialContext;
  let page = await createPage(context);
  const csvWriter = new CsvWriter();
  const listingsPerProxy = scraperRules.proxyRotation?.listingsPerProxy;

  try {
    const links = await collectListingLinks(page, CATEGORY_PATH);
    console.log(`${links.length} ilan linki bulundu.`);

    let processed = 0;
    let written = 0;
    for (const link of links) {
      try {
        const listing = await withRetry(() => scrapeListingDetail(page, link, CATEGORY_KEY));
        const isNew = await csvWriter.writeListing(listing);
        if (isNew) written += 1;
        console.log(
          `[${processed + 1}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`,
        );
      } catch (err) {
        console.error(`Ilan cekilemedi (${link}):`, err.message);
      } finally {
        await politeDelay();
        processed += 1;

        if (listingsPerProxy && processed % listingsPerProxy === 0) {
          await context.close();
          context = await createContext(browser);
          page = await createPage(context);
          console.log(`Proxy rotasyonu: ${processed} ilan sonrasi context yenilendi.`);
        }
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
