// Arabam Scraper
const { chromium } = require('playwright');
const { getProxyConfig } = require('../../config/proxy-config');
const scraperRules = require('../../config/scraper-rules.json');
const { parseListing } = require('../utils/json-parser');
const { kafka, TOPIC_RAW_LISTINGS } = require('../../config/kafka-config');

// Ilan detay sayfasindaki Turkce etiketleri json-parser'in bekledigi ham alan adlarina esler.
// İlan No kasten eslenmiyor: property-value kopyalama tooltip'i icerdigi icin kirli metin
// donduruyor; temiz ilan_id zaten URL'den (scrapeListingDetail) turetiliyor.
const DETAIL_LABEL_MAP = {
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
  'Boya-değişen': 'boyaDegisen',
  'Ağır Hasarlı': 'agirHasarli',
};

// scraper-rules.json'daki kategori anahtarlarini arac_turu alani icin okunabilir etikete cevirir.
const CATEGORY_LABELS = {
  otomobil: 'Otomobil',
  suv: 'SUV',
  minivan_panelvan: 'Minivan-Panelvan',
  elektrikli: 'Elektrikli',
};

// "1 değişen, 2 boyalı" / "1 boyalı, 1 lokal boyalı" gibi birlesik metni degisen/boyali sayilarina ayirir.
// "Belirtilmemiş" bilinmiyor anlamina gelir (null); deger var ama bir tur hic gecmiyorsa o tur icin 0 demektir
// (arabam.com sadece sifirdan farkli olan turleri listeler).
function parseBoyaDegisen(str) {
  if (!str || str === 'Belirtilmemiş') return { degisenSayisi: null, boyaliSayisi: null };

  const degisenMatch = str.match(/(\d+)\s*değişen/i);
  const boyaliMatches = [...str.matchAll(/(\d+)\s*(?:lokal\s*)?boyalı/gi)];
  const boyaliSayisi = boyaliMatches.reduce((sum, match) => sum + Number(match[1]), 0);

  return {
    degisenSayisi: degisenMatch ? Number(degisenMatch[1]) : 0,
    boyaliSayisi,
  };
}

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

// Proxy artik tarayici degil, context seviyesinde atanir (bkz. createContext) - boylece
// ayni tarayici acikken bile context'i yeniden kurarak farkli statik IP'lere gecebiliriz.
async function launchBrowser() {
  const browser = await chromium.launch({ headless: true });
  const context = await createContext(browser);
  return { browser, context };
}

// Proxy havuzundan (config/proxy-config.js) bir sonraki IP ile yeni bir context acar.
// Chromium context seviyesinde proxy override'i destekler, tek tarayiciyla rotasyon yapabiliriz.
async function createContext(browser) {
  const proxy = getProxyConfig();
  return browser.newContext({ userAgent: USER_AGENT, proxy: proxy || undefined });
}

async function createPage(context) {
  const page = await context.newPage();
  await blockUnnecessaryResources(page);
  return page;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// scraper-rules.json'daki rateLimit araligindan rastgele bir bekleme suresi (ms) hesaplar.
function randomDelay() {
  const { minDelayMs, maxDelayMs } = scraperRules.rateLimit;
  return Math.floor(Math.random() * (maxDelayMs - minDelayMs + 1)) + minDelayMs;
}

// Istekler arasi nazik bekleme: siteyi hizindan asiri yuklememek icin rastgele gecikme uygular.
async function politeDelay() {
  await sleep(randomDelay());
}

// Basarisiz olan async islemi artan bekleme sureleriyle (backoff) yeniden dener.
async function withRetry(fn, { maxAttempts, backoffMs } = scraperRules.retry) {
  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < maxAttempts) {
        await sleep(backoffMs * attempt);
      }
    }
  }
  throw lastError;
}

// Bir kategori icin liste sayfalarinda pagination ile gezip ilan detay linklerini toplar.
async function collectListingLinks(page, categoryPath) {
  const links = new Set();
  let currentUrl = `${scraperRules.baseUrl}${categoryPath}`;
  let pageCount = 0;

  while (currentUrl && pageCount < scraperRules.listPage.maxPagesPerCategory) {
    if (pageCount > 0) {
      await politeDelay();
    }
    try {
      await withRetry(() => page.goto(currentUrl, { waitUntil: 'domcontentloaded' }));

      const pageLinks = await page.$$eval(scraperRules.listPage.listingLinkSelector, (anchors) =>
        anchors.map((a) => a.href),
      );
      pageLinks.forEach((link) => links.add(link));

      currentUrl = await page
        .$eval(scraperRules.listPage.nextPageSelector, (a) => a.href)
        .catch(() => null);
    } catch (err) {
      console.error(`Sayfa ${pageCount + 1} yuklenirken hata olustu (${currentUrl}):`, err.message);
      break; // Stop collecting links for this category, but return what we have so far
    }

    pageCount += 1;
  }

  return Array.from(links);
}

// Tanimli tum kategorileri gezip her linki kesfedildigi kategoriyle birlikte doner (link -> categoryKey).
async function collectAllListingLinks(page) {
  const linkCategories = new Map();

  for (const [categoryKey, categoryPath] of Object.entries(scraperRules.categories)) {
    const categoryLinks = await collectListingLinks(page, categoryPath);
    categoryLinks.forEach((link) => {
      if (!linkCategories.has(link)) linkCategories.set(link, categoryKey);
    });
  }

  return linkCategories;
}

// Ilan detay sayfasindaki ozellik tablosunu { "Marka": "Volkswagen", ... } seklinde ham etiket-deger ciftlerine cevirir.
async function extractDetailProperties(page) {
  return page.$$eval(scraperRules.detailPage.propertiesTableSelector, (items) =>
    items.reduce((acc, item) => {
      const label = item.querySelector('.property-key')?.textContent?.trim();
      const value = item.querySelector('.property-value')?.textContent?.trim();
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
// categoryKey, ilanin hangi kategori listesinden kesfedildigini belirtir (arac_turu icin kullanilir).
async function scrapeListingDetail(page, url, categoryKey) {
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  const ilanNo = url.split('/').filter(Boolean).pop();
  const kategori = CATEGORY_LABELS[categoryKey] || null;
  const fiyat = await page
    .$eval(scraperRules.detailPage.priceSelector, (el) => el.textContent.trim())
    .catch(() => null);
  const properties = await extractDetailProperties(page).catch(() => ({}));

  const raw = mapDetailPropertiesToRaw(properties, { ilanNo, kategori, fiyat });
  const { degisenSayisi, boyaliSayisi } = parseBoyaDegisen(raw.boyaDegisen);
  return parseListing({ ...raw, degisenSayisi, boyaliSayisi });
}

// Normalize edilmis ilani Kafka raw-listings topic'ine gonderir.
async function publishListing(producer, listing) {
  await producer.send({
    topic: TOPIC_RAW_LISTINGS,
    messages: [{ key: listing.ilan_id || undefined, value: JSON.stringify(listing) }],
  });
}

// Uctan uca akis: linkleri topla, her ilani cek, normalize et, Kafka'ya gonder.
// listPage.listingLinkSelector'i toplarken kullanilan page/context sabit kalir; sadece detay
// kazima dongusunde config'deki proxyRotation.listingsPerProxy'e gore context (ve proxy) yenilenir.
async function run() {
  const { browser, context: initialContext } = await launchBrowser();
  let context = initialContext;
  let page = await createPage(context);

  const producer = kafka.producer();
  const listingsPerProxy = scraperRules.proxyRotation?.listingsPerProxy;

  await producer.connect();

  try {
    const linkCategories = await collectAllListingLinks(page);
    console.log(`${linkCategories.size} ilan linki bulundu.`);

    let processed = 0;
    for (const [link, categoryKey] of linkCategories) {
      try {
        const listing = await withRetry(() => scrapeListingDetail(page, link, categoryKey));
        await publishListing(producer, listing);
        console.log(`Yayinlandi: ${listing.ilan_id}`);
      } catch (err) {
        console.error(`Ilan cekilemedi (${link}):`, err.message);
      } finally {
        await politeDelay();
        processed += 1;

        if (listingsPerProxy && processed % listingsPerProxy === 0) {
          await context.close();
          context = await createContext(browser);
          page = await createPage(context);
        }
      }
    }
  } finally {
    await context.close();
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
  createContext,
  createPage,
  blockUnnecessaryResources,
  sleep,
  randomDelay,
  politeDelay,
  withRetry,
  collectListingLinks,
  collectAllListingLinks,
  extractDetailProperties,
  mapDetailPropertiesToRaw,
  scrapeListingDetail,
  publishListing,
  run,
};
