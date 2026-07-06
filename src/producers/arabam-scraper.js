// Arabam Scraper
const { chromium } = require('playwright');
const { getProxyConfig } = require('../../config/proxy-config');
const scraperRules = require('../../config/scraper-rules.json');

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

module.exports = {
  launchBrowser,
  createPage,
  blockUnnecessaryResources,
  collectListingLinks,
  collectAllListingLinks,
};
