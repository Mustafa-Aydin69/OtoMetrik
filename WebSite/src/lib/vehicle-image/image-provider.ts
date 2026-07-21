/**
 * Görsel kaynağı sağlayıcıları: her biri bir metin sorgusunu alıp aday
 * görsel listesi döner. Fallback sırası (bkz. getImageCandidates):
 *   1. Wikimedia Commons arama (anahtarsız, ücretsiz, çok adaylı)
 *   2. Google Custom Search (opsiyonel — GOOGLE_CSE_API_KEY/GOOGLE_CSE_CX
 *      tanımlıysa aktif olur; tanımlı değilse sessizce atlanır. Anahtarsız
 *      /ücretsiz gerçek bir Google Images API yok; ham HTML scraping ToS'a
 *      aykırı ve kırılgan olduğu için burada yok)
 *   3. Wikipedia infobox fotoğrafı (anahtarsız, tek aday, son çare)
 */
import { cached, CACHE_TTL } from "./cache";
import type { ImageCandidate } from "./types";

const COMMONS_API = "https://commons.wikimedia.org/w/api.php";
const WIKI_API = "https://en.wikipedia.org/w/api.php";
const USER_AGENT = "OtoMetrikAI/1.0 (fiyat tahmin sitesi; iletisim yok)";

const NON_PHOTO_EXTENSIONS = [".pdf", ".djvu", ".svg", ".ogv", ".webm", ".oga", ".ogg", ".tiff", ".tif"];

interface CommonsPage {
  title?: string;
  imageinfo?: Array<{
    url?: string;
    thumburl?: string;
    extmetadata?: Record<string, { value?: string }>;
  }>;
}

async function searchCommons(query: string, limit: number): Promise<ImageCandidate[]> {
  return cached(`commons:${query.toLowerCase()}:${limit}`, CACHE_TTL.IMAGE_SEARCH, async () => {
    const params = new URLSearchParams({
      action: "query",
      generator: "search",
      gsrsearch: query,
      gsrnamespace: "6", // File: namespace
      gsrlimit: String(limit),
      prop: "imageinfo",
      iiprop: "url|extmetadata",
      iiurlwidth: "800",
      format: "json",
    });

    try {
      const res = await fetch(`${COMMONS_API}?${params.toString()}`, {
        headers: { "User-Agent": USER_AGENT },
        signal: AbortSignal.timeout(6000),
      });
      if (!res.ok) return [];

      const data = (await res.json()) as { query?: { pages?: Record<string, CommonsPage> } };
      const pages = data.query?.pages ?? {};

      const candidates: ImageCandidate[] = [];
      for (const page of Object.values(pages)) {
        // Commons taranmış PDF/DjVu belge sayfalarını da "File:" isim
        // alanında tutar; OCR metni sorguyla eşleşirse bunlar araç
        // fotoğrafı gibi görünen ama alakasız sonuçlar olarak sızabilir
        // (örn. tip onayı PDF'leri). Gerçek fotoğraf olmayan dosya
        // uzantılarını burada eliyoruz.
        const title = page.title ?? "";
        if (NON_PHOTO_EXTENSIONS.some((ext) => title.toLowerCase().endsWith(ext))) continue;

        const info = page.imageinfo?.[0];
        if (!info) continue;
        const url = info.thumburl ?? info.url;
        if (!url) continue;

        const meta = info.extmetadata ?? {};
        const sourceText = [
          page.title,
          meta.ObjectName?.value,
          meta.ImageDescription?.value,
          meta.Categories?.value,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        candidates.push({ url, provider: "commons", sourceText });
      }
      return candidates;
    } catch {
      return [];
    }
  });
}

interface GoogleCseItem {
  link?: string;
  title?: string;
  snippet?: string;
}

// Opsiyonel katman: GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX ortam değişkenleri
// tanımlıysa Google Custom Search JSON API'sini (searchType=image) kullanır.
// Tanımlı değilse (varsayılan) boş liste döner — özellik sessizce devre
// dışı kalır, aynı projedeki PREDICTION_API_URL/proxy deseninde olduğu gibi.
async function searchGoogleCse(query: string, limit: number): Promise<ImageCandidate[]> {
  const apiKey = process.env.GOOGLE_CSE_API_KEY;
  const cx = process.env.GOOGLE_CSE_CX;
  if (!apiKey || !cx) return [];

  return cached(`google-cse:${query.toLowerCase()}:${limit}`, CACHE_TTL.IMAGE_SEARCH, async () => {
    const params = new URLSearchParams({
      key: apiKey,
      cx,
      q: query,
      searchType: "image",
      num: String(Math.min(limit, 10)),
      safe: "active",
    });

    try {
      const res = await fetch(`https://www.googleapis.com/customsearch/v1?${params.toString()}`, {
        signal: AbortSignal.timeout(6000),
      });
      if (!res.ok) return [];

      const data = (await res.json()) as { items?: GoogleCseItem[] };
      return (data.items ?? [])
        .filter((item): item is GoogleCseItem & { link: string } => !!item.link)
        .map((item) => ({
          url: item.link,
          provider: "google-cse" as const,
          sourceText: [item.title, item.snippet].filter(Boolean).join(" ").toLowerCase(),
        }));
    } catch {
      return [];
    }
  });
}

interface WikiPage {
  title?: string;
  thumbnail?: { source?: string };
}

// Son çare: Wikipedia makalesinin infobox fotoğrafı (tek aday, sadece
// marka+model'e göre — nesil/renk/paket filtrelemez).
async function fetchWikipediaInfobox(brand: string, model: string): Promise<ImageCandidate[]> {
  const title = `${brand} ${model}`.trim();
  if (!title) return [];

  const params = new URLSearchParams({
    action: "query",
    titles: title,
    prop: "pageimages",
    format: "json",
    pithumbsize: "800",
    redirects: "1",
  });

  try {
    const res = await fetch(`${WIKI_API}?${params.toString()}`, {
      headers: { "User-Agent": USER_AGENT },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];

    const data = (await res.json()) as { query?: { pages?: Record<string, WikiPage> } };
    const pages = data.query?.pages;
    if (!pages) return [];

    const page = Object.values(pages)[0];
    const url = page?.thumbnail?.source;
    if (!url) return [];

    return [{ url, provider: "wikipedia-infobox", sourceText: (page?.title ?? title).toLowerCase() }];
  } catch {
    return [];
  }
}

/**
 * queryVariants (en spesifikten en genele) üzerinde Commons + Google CSE'yi
 * dener, ilk sonuç veren varyantta durur. Hiçbiri sonuç vermezse Wikipedia
 * infobox'a düşer. En fazla `limit` aday döner.
 */
export async function getImageCandidates(
  brand: string,
  model: string,
  queryVariants: string[],
  limit = 10
): Promise<ImageCandidate[]> {
  for (const query of queryVariants) {
    const [commons, google] = await Promise.all([
      searchCommons(query, limit),
      searchGoogleCse(query, limit),
    ]);
    const combined = [...commons, ...google];
    if (combined.length > 0) return combined.slice(0, limit);
  }

  return fetchWikipediaInfobox(brand, model);
}
