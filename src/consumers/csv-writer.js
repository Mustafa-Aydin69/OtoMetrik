// CSV Writer
const fs = require('fs');
const path = require('path');
const { createObjectCsvWriter } = require('csv-writer');

const CSV_PATH = path.join(__dirname, '../../data/output/arabam_test_val.csv');

const SCHEMA_FIELDS = [
  'ilan_id',
  'arac_turu',
  'marka',
  'model',
  'paket',
  'kasa_turu',
  'renk',
  'motor_hacmi',
  'motor_gucu',
  'yil',
  'kilometre',
  'yakit_turu',
  'vites',
  'degisen_sayisi',
  'boyali_sayisi',
  'agir_hasarli',
  'fiyat',
  'scraped_at',
];

// Var olan CSV'den daha once yazilmis ilan_id'leri okuyup duplike kontrolu icin bir kume dondurur.
function loadExistingIds(csvPath) {
  if (!fs.existsSync(csvPath)) return new Set();
  const content = fs.readFileSync(csvPath, 'utf-8').trim();
  if (!content) return new Set();

  const [, ...lines] = content.split('\n');
  const ilanIdIndex = SCHEMA_FIELDS.indexOf('ilan_id');
  return new Set(lines.filter(Boolean).map((line) => line.split(',')[ilanIdIndex]));
}

class CsvWriter {
  constructor(csvPath = CSV_PATH) {
    this.csvPath = csvPath;
    fs.mkdirSync(path.dirname(csvPath), { recursive: true });
    this.seenIds = loadExistingIds(csvPath);
    this.writer = createObjectCsvWriter({
      path: csvPath,
      header: SCHEMA_FIELDS.map((id) => ({ id, title: id })),
      append: fs.existsSync(csvPath),
    });
  }

  // Bir ilani semaya gore CSV'ye ekler; ilan_id daha once yazilmissa atlar.
  async writeListing(listing) {
    const id = listing.ilan_id;
    if (id && this.seenIds.has(id)) return false;

    await this.writer.writeRecords([listing]);
    if (id) this.seenIds.add(id);
    return true;
  }
}

module.exports = { CsvWriter, SCHEMA_FIELDS, CSV_PATH };
