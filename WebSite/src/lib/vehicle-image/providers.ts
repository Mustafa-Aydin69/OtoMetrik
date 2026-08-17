/**
 * Görsel kaynağı sağlayıcıları (Candidate Retrieval katmanı) - her biri
 * VehicleImageProvider arayüzünü uygular (bkz. types.ts), yalnızca aday
 * DÖNER, hiçbir değerlendirme/seçim yapmaz (bkz. plan Madde 8 - "search
 * engine'ın ilk sonucu nihai sonuç kabul edilmemeli"). Değerlendirme
 * candidate-ranker.ts + confidence.ts'te.
 *
 * Sağlayıcılar:
 *   1. commonsProvider  - Wikimedia Commons arama (anahtarsız, ücretsiz, çok
 *      adaylı). Commons'ın extmetadata.Categories alanı "|" ile ayrılmış
 *      kategori listesi döner (canlı doğrulandı, örn. "BMW G20 (2022)|...|
 *      BMW 320i|BMW G20 in police service") - bu, candidate-ranker.ts'in hard
 *      conflict tespiti için en güçlü sinyal (bkz. plan Madde 9/10).
 *   2. googleCseProvider - opsiyonel (GOOGLE_CSE_API_KEY/GOOGLE_CSE_CX
 *      tanımlıysa aktif olur, aksi halde sessizce atlanır).
 *   3. fetchWikipediaInfobox - "search" değil, tek adaylı son çare (marka+
 *      model makalesinin infobox fotoğrafı) - queryVariants tükenince
 *      vehicle-image.service.ts tarafından ayrıca çağrılır.
 *
 * İleride yeni bir kaynak (örn. üretici basın görselleri) VehicleImageProvider
 * arayüzünü uygulayıp PROVIDERS dizisine eklenebilir - orkestrasyon katmanı
 * değişmeden kalır (bkz. plan Madde 24/25 - şu an yeni provider YAZILMIYOR,
 * yalnızca abstraction hazırlanıyor).
 */
import { cached, CACHE_TTL } from "./cache";
import type { ImageCandidate, VehicleImageProvider } from "./types";

const COMMONS_API = "https://commons.wikimedia.org/w/api.php";
const WIKI_API = "https://en.wikipedia.org/w/api.php";
const USER_AGENT = "OtoMetrikAI/1.0 (fiyat tahmin sitesi; iletisim yok)";

const NON_PHOTO_EXTENSIONS = [".pdf", ".djvu", ".svg", ".ogv", ".webm", ".oga", ".ogg", ".tiff", ".tif"];

function parseCategories(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split("|")
    .map((c) => c.trim())
    .filter(Boolean);
}

/** "File:Romanian Police BMW 3 Series G20 LCI 1.jpg" -> "Romanian Police BMW 3 Series G20 LCI 1". */
function filenameFromTitle(title: string): string {
  return title.replace(/^File:/, "").replace(/\.[a-zA-Z0-9]+$/, "");
}

interface CommonsPage {
  title?: string;
  imageinfo?: Array<{
    url?: string;
    thumburl?: string;
    descriptionurl?: string;
    extmetadata?: Record<string, { value?: string }>;
  }>;
}

async function searchCommonsRaw(query: string, limit: number): Promise<ImageCandidate[]> {
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
        // alanında tutar; bunlar araç fotoğrafı gibi görünen ama alakasız
        // sonuçlar olarak sızabilir (örn. tip onayı PDF'leri). Gerçek
        // fotoğraf olmayan dosya uzantılarını burada eliyoruz.
        const title = page.title ?? "";
        if (NON_PHOTO_EXTENSIONS.some((ext) => title.toLowerCase().endsWith(ext))) continue;

        const info = page.imageinfo?.[0];
        if (!info) continue;
        const url = info.thumburl ?? info.url;
        if (!url) continue;

        const meta = info.extmetadata ?? {};
        candidates.push({
          url,
          pageUrl: info.descriptionurl,
          filename: filenameFromTitle(title),
          title: meta.ObjectName?.value ?? filenameFromTitle(title),
          description: meta.ImageDescription?.value,
          categories: parseCategories(meta.Categories?.value),
          source: "commons",
        });
      }
      return candidates;
    } catch {
      return [];
    }
  });
}

export const commonsProvider: VehicleImageProvider = {
  name: "commons",
  search: searchCommonsRaw,
};

interface GoogleCseItem {
  link?: string;
  title?: string;
  snippet?: string;
}

// Opsiyonel katman: GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX ortam değişkenleri
// tanımlıysa Google Custom Search JSON API'sini (searchType=image) kullanır.
// Tanımlı değilse (varsayılan) boş liste döner - özellik sessizce devre
// dışı kalır, aynı projedeki PREDICTION_API_URL/proxy deseninde olduğu gibi.
async function searchGoogleCseRaw(query: string, limit: number): Promise<ImageCandidate[]> {
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
          title: item.title,
          description: item.snippet,
          source: "google-cse" as const,
        }));
    } catch {
      return [];
    }
  });
}

export const googleCseProvider: VehicleImageProvider = {
  name: "google-cse",
  search: searchGoogleCseRaw,
};

/** Aktif sağlayıcılar - vehicle-image.service.ts her query rung'unda hepsini paralel dener. */
export const PROVIDERS: VehicleImageProvider[] = [commonsProvider, googleCseProvider];

interface WikiPage {
  title?: string;
  thumbnail?: { source?: string };
}

/**
 * Son çare: Wikipedia makalesinin infobox fotoğrafı (tek aday, sadece
 * marka+model'e göre - nesil/renk/paket filtrelemez, bu yüzden yalnızca
 * candidate-ranker.ts'in confident bulduğu hiçbir aday yoksa denenir).
 */
export async function fetchWikipediaInfobox(make: string, model: string): Promise<ImageCandidate[]> {
  const title = `${make} ${model}`.trim();
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

    return [{ url, title: page?.title ?? title, source: "wikipedia-infobox" }];
  } catch {
    return [];
  }
}
