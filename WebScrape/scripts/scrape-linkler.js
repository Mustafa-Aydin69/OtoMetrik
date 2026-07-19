// linkler.txt'deki linkleri sirayla isler: her link icin kategori sayfasini "?take=50" ile
// gezip ilanlari data/output/arabam_test_val.csv'ye yazar (CsvWriter.seenIds ile ayni resume
// garantisi - agdan cekilmeden atlanir). Bir link tamamen islendiginde linkler.txt'deki o satir
// "// " ile yorum satirina alinir (bir sonraki calistirmada atlanir), ve logs/suv-scrape-log.txt
// dosyasina o link icin islenen/yeni ilan sayisi ile yeni ilan_id'ler eklenir.
// Bu script kesintiye ugrarsa (Ctrl+C, hata vb.) kaldigi yerden devam eder: linkler.txt'de hala
// yorumsuz olan ilk link tekrar okunur.
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { CsvWriter } = require('../src/consumers/csv-writer');
const {
  launchBrowser,
  createPage,
  collectListingLinks,
  scrapeListingDetail,
  withRetry,
  politeDelay,
} = require('../src/producers/arabam-scraper');

const CATEGORY_KEY = 'suv';
const LINKLER_PATH = path.join(__dirname, '../linkler.txt');
const LOG_PATH = path.join(__dirname, '../logs/suv-scrape-log.txt');

function isCommented(line) {
  const trimmed = line.trim();
  return trimmed.startsWith('//') || trimmed.startsWith('#');
}

// linkler.txt'yi CRLF'i koruyarak satirlara ayirir.
function readLines() {
  const raw = fs.readFileSync(LINKLER_PATH, 'utf-8');
  return raw.split(/\r\n|\n/);
}

function writeLines(lines) {
  fs.writeFileSync(LINKLER_PATH, lines.join('\r\n'));
}

function firstPendingLink() {
  const lines = readLines();
  const line = lines.find((l) => l.trim() && !isCommented(l));
  return line ? line.trim() : null;
}

// Isi biten linki linkler.txt icinde yorum satirina cevirir.
function markLinkDone(url) {
  const lines = readLines();
  const idx = lines.findIndex((l) => l.trim() === url && !isCommented(l));
  if (idx !== -1) {
    lines[idx] = `// ${lines[idx]}`;
    writeLines(lines);
  }
}

function appendLog(line) {
  fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
  fs.appendFileSync(LOG_PATH, `${line}\r\n`);
}

// Tam URL'i (baseUrl dahil) collectListingLinks'in bekledigi kategori yoluna (+ take=50) cevirir.
function toCategoryPath(url) {
  const withoutBase = url.replace(/^https?:\/\/[^/]+/, '');
  const separator = withoutBase.includes('?') ? '&' : '?';
  return `${withoutBase}${separator}take=50`;
}

async function scrapeUrl(page, csvWriter, url) {
  const categoryPath = toCategoryPath(url);
  const links = await collectListingLinks(page, categoryPath);
  console.log(`\n=== ${url}: ${links.length} ilan linki bulundu ===`);

  let processed = 0;
  let written = 0;
  const newIlanIds = [];

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
      if (isNew) {
        written += 1;
        newIlanIds.push(listing.ilan_id);
      }
      console.log(`[${processed}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`);
    } catch (err) {
      console.error(`Ilan cekilemedi (${link}):`, err.message);
    } finally {
      await politeDelay();
    }
  }

  return { processed, written, newIlanIds };
}

async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);
  const csvWriter = new CsvWriter();

  try {
    let url = firstPendingLink();
    while (url) {
      const startedAt = new Date().toISOString();
      console.log(`\n>>> Isleniyor: ${url}`);

      let result;
      try {
        result = await scrapeUrl(page, csvWriter, url);
      } catch (err) {
        console.error(`Link islenemedi, yorum satirina alinmadan birakiliyor (${url}):`, err.message);
        appendLog(`[${startedAt}] HATA ${url}: ${err.message}`);
        break; // Ayni linkte sonsuz donguye girmemek icin dur; linkler.txt'de yorumsuz kalir.
      }

      markLinkDone(url);
      const finishedAt = new Date().toISOString();
      const idsPart = result.newIlanIds.length ? ` | ilan_id: ${result.newIlanIds.join(', ')}` : '';
      appendLog(
        `[${startedAt} - ${finishedAt}] ${url} | islenen: ${result.processed}, yeni: ${result.written}${idsPart}`,
      );

      await politeDelay();
      url = firstPendingLink();
    }
    console.log('\nTum linkler islendi (veya bekleyen link kalmadi).');
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error('linkler.txt bazli kazima basarisiz:', err);
  process.exit(1);
});
