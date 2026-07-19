// linkler.txt'deki linkleri sirayla isler: her link icin kategori sayfasini "?take=50" ile
// gezip ilanlari data/output/arabam_test_val.csv'ye yazar (CsvWriter.seenIds ile ayni resume
// garantisi - agdan cekilmeden atlanir). Bir link tamamen islendiginde linkler.txt'deki o satir
// "// " ile yorum satirina alinir (bir sonraki calistirmada atlanir). Log ciktisi, diger toplu
// kazima scriptleriyle (scrape-minivan-panelvan-brands.js) ayni bicimde loglar/scrape-log-tum-markalar.txt
// dosyasina eklenir (append) - marka basi ilan satirlari, "tamamlandi" ozeti ve calisma sonunda OZET.
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
const LOG_PATH = path.join(__dirname, '../loglar/scrape-log-tum-markalar.txt');

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

// scrape-minivan-panelvan-brands.js ile ayni bicimde: console'a yazar ve ayni satiri
// loglar/scrape-log-tum-markalar.txt dosyasina ekler (append).
function log(line) {
  fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
  fs.appendFileSync(LOG_PATH, `${line}\r\n`);
  console.log(line);
}

// Tam URL'i (baseUrl dahil) collectListingLinks'in bekledigi kategori yoluna (+ take=50) cevirir.
function toCategoryPath(url) {
  const withoutBase = url.replace(/^https?:\/\/[^/]+/, '');
  const separator = withoutBase.includes('?') ? '&' : '?';
  return `${withoutBase}${separator}take=50`;
}

// URL'den marka adini ve varsa minYear/maxYear'i scrape-minivan-panelvan-brands.js'deki
// sliceLabel bicimiyle etiketler (orn. "nissan (-2015)", "peugeot (2020+)").
function urlLabel(url) {
  const [pathPart, queryPart] = url.split('?');
  const brand = pathPart.split('/').filter(Boolean).pop();
  const params = new URLSearchParams(queryPart || '');
  const minYear = params.get('minYear');
  const maxYear = params.get('maxYear');
  if (minYear && maxYear) return `${brand} (${minYear}-${maxYear})`;
  if (minYear) return `${brand} (${minYear}+)`;
  if (maxYear) return `${brand} (-${maxYear})`;
  return brand;
}

async function scrapeUrl(page, csvWriter, url) {
  const label = urlLabel(url);
  const categoryPath = toCategoryPath(url);
  const links = await collectListingLinks(page, categoryPath);
  log(`\n=== ${label}: ${links.length} ilan linki bulundu ===`);

  let processed = 0;
  let written = 0;

  for (const link of links) {
    processed += 1;
    const ilanId = link.split('/').filter(Boolean).pop();
    if (csvWriter.seenIds.has(ilanId)) {
      log(`[${label} ${processed}/${links.length}] ${ilanId} zaten vardi (atlandi, cekilmedi)`);
      continue;
    }

    try {
      const listing = await withRetry(() => scrapeListingDetail(page, link, CATEGORY_KEY));
      const isNew = await csvWriter.writeListing(listing);
      if (isNew) written += 1;
      log(`[${label} ${processed}/${links.length}] ${listing.ilan_id} ${isNew ? 'yazildi' : 'zaten vardi (atlandi)'}`);
    } catch (err) {
      console.error(`Ilan cekilemedi (${link}):`, err.message);
    } finally {
      await politeDelay();
    }
  }

  log(`--- ${label} tamamlandi: ${written} yeni kayit (${processed} ilan islendi) ---`);
  return { label, processed, written };
}

async function run() {
  const { browser, context } = await launchBrowser();
  const page = await createPage(context);
  const csvWriter = new CsvWriter();

  const results = [];
  try {
    let url = firstPendingLink();
    while (url) {
      console.log(`\n>>> Isleniyor: ${url}`);

      let result;
      try {
        result = await scrapeUrl(page, csvWriter, url);
      } catch (err) {
        console.error(`Link islenemedi, yorum satirina alinmadan birakiliyor (${url}):`, err.message);
        break; // Ayni linkte sonsuz donguye girmemek icin dur; linkler.txt'de yorumsuz kalir.
      }

      results.push(result);
      markLinkDone(url);
      await politeDelay();
      url = firstPendingLink();
    }
    console.log('\nTum linkler islendi (veya bekleyen link kalmadi).');
  } finally {
    await context.close();
    await browser.close();
  }

  if (results.length) {
    const totalWritten = results.reduce((sum, r) => sum + r.written, 0);
    const totalProcessed = results.reduce((sum, r) => sum + r.processed, 0);
    log('\n=== OZET ===');
    for (const r of results) {
      log(`${r.label}: ${r.written} yeni / ${r.processed} islendi`);
    }
    log(`TOPLAM: ${totalWritten} yeni kayit yazildi (${totalProcessed} ilan islendi).`);
  }
}

run().catch((err) => {
  console.error('linkler.txt bazli kazima basarisiz:', err);
  process.exit(1);
});
