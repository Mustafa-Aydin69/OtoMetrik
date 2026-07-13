// JSON Parser
const { normalizeWhitespace, parsePriceTL, parseKilometre } = require('./text-cleaner');

const ENGINE_CODE_PATTERN = '(?:TSI|TDI|TFSI|CRDI|VTI|DCI|HDI|CDTI|MPI|CGI|BLUEHDI|ECOBOOST)';
const MODEL_DETAIL_REGEX = new RegExp(`^(\\d+(?:\\.\\d+)?(?:\\s*${ENGINE_CODE_PATTERN})?)\\s*(.*)$`, 'i');

// Ham ilan başlığındaki "1.5 TSI R-Line" gibi birleşik metni motor hacmi ve paket olarak ayırır.
function splitModelDetail(modelDetail) {
  const str = normalizeWhitespace(modelDetail || '');
  if (!str) return { motorHacmi: null, paket: null };
  const match = str.match(MODEL_DETAIL_REGEX);
  if (!match) return { motorHacmi: null, paket: str };
  return {
    motorHacmi: match[1].trim() || null,
    paket: match[2].trim() || null,
  };
}

// arabam.com "Ağır Hasarlı" alanı "Evet"/"Hayır" doner; ortak semada digerleriyle (degisen_sayisi vb.)
// tutarli olmasi icin 1/0'a ceviriyoruz. Deger yoksa/tanimadigimiz bir seyse null (bilinmiyor).
function parseEvetHayir(value) {
  if (value == null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'evet') return 1;
  if (normalized === 'hayır' || normalized === 'hayir') return 0;
  return null;
}

function parseListing(raw) {
  const { motorHacmi, paket } = splitModelDetail(raw.model);

  return {
    ilan_id: raw.ilanNo != null ? String(raw.ilanNo) : null,
    arac_turu: raw.kategori || null,
    marka: raw.marka || null,
    model: raw.seri || null,
    paket: raw.paket || paket,
    kasa_turu: raw.kasaTipi || null,
    renk: raw.renk || null,
    motor_hacmi: raw.motorHacmi || motorHacmi,
    motor_gucu: raw.motorGucu != null ? String(raw.motorGucu) : null,
    yil: raw.yil != null ? Number(raw.yil) : null,
    kilometre: parseKilometre(raw.kilometre),
    yakit_turu: raw.yakitTipi || null,
    vites: raw.vitesTipi || null,
    degisen_sayisi: raw.degisenSayisi != null ? Number(raw.degisenSayisi) : null,
    boyali_sayisi: raw.boyaliSayisi != null ? Number(raw.boyaliSayisi) : null,
    // arabam.com "Agir Hasarli" satirini sadece "Evet" oldugunda gosteriyor (Boya-degisen'deki
    // "sifirdan farkli olani listele" konvansiyonuyla ayni); sayfada hic yoksa "Hayir" demektir.
    agir_hasarli: raw.agirHasarli !== undefined ? parseEvetHayir(raw.agirHasarli) : 0,
    fiyat: parsePriceTL(raw.fiyat),
    scraped_at: new Date().toISOString(),
  };
}

module.exports = { parseListing, splitModelDetail };
