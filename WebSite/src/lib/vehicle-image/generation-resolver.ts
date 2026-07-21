/**
 * Marka+model+yıldan, Wikipedia'nın kendi sayfa yapısını okuyarak dinamik
 * olarak nesil (generation) etiketi çıkarır — hiçbir marka/model/nesil
 * hardcoded değildir.
 *
 * Wikipedia'daki araç makaleleri neredeyse hep şu kalıbı kullanır:
 *   "First generation (E21; 1975)"
 *   "Seventh generation (G20/G21/G28; 2018)"
 *   "Twelfth generation (E210; 2018)"
 * yani "{Sıra} generation ({kod(lar)}; {başlangıç yılı})". Bu fonksiyon
 * action=parse&prop=sections ile başlık listesini çeker, bu kalıba uyan
 * satırları ayrıştırır, her nesle bir [başlangıç, bitiş) yıl aralığı
 * atar (bitiş = bir sonraki neslin başlangıcı, son nesil için "şimdi") ve
 * hedef yılın düştüğü nesli döner.
 *
 * Bilinen sınırlama: Bazı markalar (örn. Ford) kendi pazarlama isimlerini
 * ("Mk7") Wikipedia'nın "generation" başlıklarında KULLANMAZ — makale
 * "Sixth generation (B299/B409; 2008)" yazar, "Mk7" yazmaz. Bu durumda
 * dönen etiket "B299" olur, "Mk7" değil. Bunu marka bazlı bir çeviri
 * tablosuyla "düzeltmek" hardcoding'e geri döner, o yüzden kasıtlı olarak
 * yapılmıyor — etiket doğrudan Wikipedia'nın kendi verisinden türetiliyor.
 */
import { cached, CACHE_TTL } from "./cache";
import type { ResolvedGeneration } from "./types";

const WIKI_API = "https://en.wikipedia.org/w/api.php";

interface GenerationTimelineEntry {
  ordinalLabel: string;
  codes: string[];
  startYear: number;
}

interface WikiSection {
  line?: string;
}

async function fetchSections(pageTitle: string): Promise<WikiSection[]> {
  const params = new URLSearchParams({
    action: "parse",
    page: pageTitle,
    prop: "sections",
    format: "json",
    redirects: "1",
  });

  const res = await fetch(`${WIKI_API}?${params.toString()}`, {
    headers: { "User-Agent": "OtoMetrikAI/1.0 (fiyat tahmin sitesi; iletisim yok)" },
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) return [];

  const data = (await res.json()) as {
    parse?: { sections?: WikiSection[] };
    error?: unknown;
  };
  if (data.error) return [];
  return data.parse?.sections ?? [];
}

// "Seventh generation (G20/G21/G28; 2018)" -> { ordinalLabel: "Seventh generation", codes: ["G20","G21","G28"], startYear: 2018 }
// Parantezsiz ya da yıl içermeyen başlıklar (örn. "Motorsport") atlanır.
function parseGenerationHeading(line: string): GenerationTimelineEntry | null {
  const m = line.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
  if (!m) return null;

  const ordinalLabel = m[1].trim();
  const parenContent = m[2];
  const semiIdx = parenContent.lastIndexOf(";");
  if (semiIdx === -1) return null;

  const codesPart = parenContent.slice(0, semiIdx).trim();
  const yearPart = parenContent.slice(semiIdx + 1).trim();
  const yearMatch = yearPart.match(/^(\d{4})/);
  if (!yearMatch || !codesPart) return null;

  const codes = codesPart
    .split(/[/,&]/)
    .map((c) => c.trim())
    .filter(Boolean);
  if (codes.length === 0) return null;

  return { ordinalLabel, codes, startYear: Number(yearMatch[1]) };
}

async function fetchGenerationTimeline(pageTitle: string): Promise<GenerationTimelineEntry[]> {
  return cached(`generation-timeline:${pageTitle.toLowerCase()}`, CACHE_TTL.GENERATION, async () => {
    const sections = await fetchSections(pageTitle);
    const entries: GenerationTimelineEntry[] = [];
    for (const section of sections) {
      if (!section.line) continue;
      const parsed = parseGenerationHeading(section.line);
      if (parsed) entries.push(parsed);
    }
    entries.sort((a, b) => a.startYear - b.startYear);
    return entries;
  });
}

/**
 * brand+model'in Wikipedia makalesinden, verilen yıla düşen nesli çözer.
 * Yıl verilmemişse veya makale nesil başlıkları içermiyorsa null döner —
 * pipeline generation olmadan da (daha az spesifik bir sorguyla) çalışmaya
 * devam eder.
 */
export async function resolveGeneration(
  brand: string,
  model: string,
  year: number | null
): Promise<ResolvedGeneration | null> {
  if (!year) return null;

  const pageTitle = `${brand} ${model}`.trim();
  if (!pageTitle) return null;

  let timeline: GenerationTimelineEntry[];
  try {
    timeline = await fetchGenerationTimeline(pageTitle);
  } catch {
    return null;
  }
  if (timeline.length === 0) return null;

  // Hedef yıldan önce/eşit başlayan en son nesil (liste yıla göre artan sıralı).
  let match: GenerationTimelineEntry | null = null;
  for (const entry of timeline) {
    if (entry.startYear <= year) {
      match = entry;
    } else {
      break;
    }
  }
  if (!match) return null;

  return {
    label: match.codes[0],
    ordinalLabel: match.ordinalLabel,
    allCodes: match.codes,
  };
}
