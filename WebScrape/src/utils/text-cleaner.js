// Text Cleaner
function normalizeWhitespace(str) {
  return String(str).trim().replace(/\s+/g, ' ');
}

function parseNumber(str) {
  const digits = String(str).replace(/[^\d]/g, '');
  return digits ? parseInt(digits, 10) : null;
}

function parsePriceTL(str) {
  return parseNumber(str);
}

function parseKilometre(str) {
  return parseNumber(str);
}

module.exports = { normalizeWhitespace, parseNumber, parsePriceTL, parseKilometre };
