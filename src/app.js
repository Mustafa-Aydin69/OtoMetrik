// Main Application Entry Point
require('dotenv').config();
const { CsvWriter } = require('./consumers/csv-writer');
const { consumeListings } = require('./consumers/kafka-consumer');
const { run: runScraper } = require('./producers/arabam-scraper');

// Consumer'i baslatir (CSV'ye yazmaya baslar), ardindan scraper'i (producer) calistirir;
// scraper bitince consumer'i kapatir.
async function main() {
  const csvWriter = new CsvWriter();
  const consumer = await consumeListings((listing) => csvWriter.writeListing(listing));

  try {
    await runScraper();
  } finally {
    await consumer.disconnect();
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error('Uygulama basarisiz:', err);
    process.exit(1);
  });
}

module.exports = { main };
