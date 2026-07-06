// Arabam Scraper
const { chromium } = require('playwright');
const { getProxyConfig } = require('../../config/proxy-config');
const scraperRules = require('../../config/scraper-rules.json');
const { parseListing } = require('../utils/json-parser');
const { kafka, TOPIC_RAW_LISTINGS } = require('../../config/kafka-config');

// Ilan detay sayfasindaki Turkce etiketleri json-parser'in bekledigi ham alan adlarina esler.
const DETAIL_LABEL_MAP = {
  'İlan No': 'ilanNo',
  Marka: 'marka',
  Seri: 'seri',
  Model: 'model',
  Yıl: 'yil',
  Kilometre: 'kilometre',
  'Vites Tipi': 'vitesTipi',
  'Yakıt Tipi': 'yakitTipi',
  'Kasa Tipi': 'kasaTipi',
  Renk: 'renk',
  'Motor Hacmi': 'motorHacmi',
  'Motor Gücü': 'motorGucu',
  Boyalı: 'boyaliSayisi',
  Değişen: 'degisenSayisi',
};

const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

// Gereksiz kaynaklari (resim/font/medya/reklam) engelleyerek trafik ve hizi optimize eder.
async function blockUnnecessaryResources(page) {
  await page.route('**/*', (route) => {
    const request = route.request();
    const resourceType = request.resourceType();
    const url = request.url();

    if (scraperRules.blockedResourceTypes.includes(resourceType)) {
      return route.abort();
    }

    if (scraperRules.blockedDomains.some((domain) => url.includes(domain))) {
      return route.abort();
    }

    return route.continue();
  });
}

async function launchBrowser() {
  const proxy = getProxyConfig();
  const browser = await chromium.launch({
    headless: true,
    proxy: proxy || undefined,
  });

  const context = await browser.newContext({ userAgent: USER_AGENT });
  return { browser, context };
}

async function createPage(context) {
  const page = await context.newPage();
  await blockUnnecessaryResources(page);
  return page;
}

// Bir kategori icin liste sayfalarinda pagination ile gezip ilan detay linklerini toplar.
async function collectListingLinks(page, categoryPath) {
  const links = new Set();
  let currentUrl = `${scraperRules.baseUrl}${categoryPath}`;
  let pageCount = 0;

  while (currentUrl && pageCount < scraperRules.listPage.maxPagesPerCategory) {
    await page.goto(currentUrl, { waitUntil: 'domcontentloaded' });

    const pageLinks = await page.$$eval(scraperRules.listPage.listingLinkSelector, (anchors) =>
      anchors.map((a) => a.href),
    );
    pageLinks.forEach((link) => links.add(link));

    currentUrl = await page
      .$eval(scraperRules.listPage.nextPageSelector, (a) => a.href)
      .catch(() => null);

    pageCount += 1;
  }

  return Array.from(links);
}

// Tanimli tum kategorileri gezip essiz ilan linklerinin birlesik listesini doner.
async function collectAllListingLinks(page) {
  const allLinks = new Set();

  for (const categoryPath of Object.values(scraperRules.categories)) {
    const categoryLinks = await collectListingLinks(page, categoryPath);
    categoryLinks.forEach((link) => allLinks.add(link));
  }

  return Array.from(allLinks);
}

// Ilan detay sayfasindaki ozellik tablosunu { "Marka": "Volkswagen", ... } seklinde ham etiket-deger ciftlerine cevirir.
async function extractDetailProperties(page) {
  return page.$$eval(`${scraperRules.detailPage.propertiesTableSelector} li`, (items) =>
    items.reduce((acc, item) => {
      const label = item.querySelector(':first-child')?.textContent?.trim();
      const value = item.querySelector(':last-child')?.textContent?.trim();
      if (label && value) acc[label] = value;
      return acc;
    }, {}),
  );
}

function mapDetailPropertiesToRaw(properties, extra) {
  const raw = { ...extra };
  for (const [label, value] of Object.entries(properties)) {
    const key = DETAIL_LABEL_MAP[label];
    if (key) raw[key] = value;
  }
  return raw;
}

// Tek bir ilan detay sayfasini cekip json-parser ile 17 alanli semaya normalize eder.
async function scrapeListingDetail(page, url) {
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  const ilanNo = url.split('/').filter(Boolean).pop();
  const kategori = await page
    .$eval(scraperRules.detailPage.titleSelector, (el) => el.textContent.trim())
    .catch(() => null);
  const fiyat = await page
    .$eval(scraperRules.detailPage.priceSelector, (el) => el.textContent.trim())
    .catch(() => null);
  const properties = await extractDetailProperties(page).catch(() => ({}));

  const raw = mapDetailPropertiesToRaw(properties, { ilanNo, kategori, fiyat });
  return parseListing(raw);
}

// Normalize edilmis ilani Kafka raw-listings topic'ine gonderir.
async function publishListing(producer, listing) {
  await producer.send({
    topic: TOPIC_RAW_LISTINGS,
    messages: [{ key: listing.ilan_id || undefined, value: JSON.stringify(listing) }],
  });
}

// Uctan uca akis: linkleri topla, her ilani cek, normalize et, Kafka'ya gonder.
async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);

  const producer = kafka.producer();
  await producer.connect();

  try {
    const links = await collectAllListingLinks(page);
    console.log(`${links.length} ilan linki bulundu.`);

    for (const link of links) {
      try {
        const listing = await scrapeListingDetail(page, link);
        await publishListing(producer, listing);
        console.log(`Yayinlandi: ${listing.ilan_id}`);
      } catch (err) {
        console.error(`Ilan cekilemedi (${link}):`, err.message);
      }
    }
  } finally {
    await producer.disconnect();
    await browser.close();
  }
}

if (require.main === module) {
  run().catch((err) => {
    console.error('Scraper basarisiz:', err);
    process.exit(1);
  });
}

module.exports = {
  launchBrowser,
  createPage,
  blockUnnecessaryResources,
  collectListingLinks,
  collectAllListingLinks,
  extractDetailProperties,
  mapDetailPropertiesToRaw,
  scrapeListingDetail,
  publishListing,
  run,
};
