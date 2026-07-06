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

module.exports = { launchBrowser, createPage, blockUnnecessaryResources };
