/**
 * Araç görsel bulma pipeline'ının katmanlar arası paylaşılan tipleri.
 * Bkz. vehicle-image.service.ts için genel akış.
 */

export interface NormalizedVehicle {
  brand: string;
  model: string;
  /** null ise generation-resolver.ts tarafından doldurulmaya çalışılır. */
  generation: string | null;
  year: number | null;
  trim: string | null;
  /** İngilizce (arama motorları/Commons için) — vehicle-parser.ts çevirir. */
  color: string | null;
}

export interface ResolvedGeneration {
  /** Aranabilir kısa etiket — Wikipedia'nın parantez içi ilk kodu (örn. "G20", "E210", "B299"). */
  label: string;
  /** Wikipedia'nın sıra sayısı ifadesi (örn. "Seventh generation") — ikincil arama varyantı için. */
  ordinalLabel: string;
  /** Parantez içindeki tüm kodlar ("/" ile ayrılmış), label bunların ilki. */
  allCodes: string[];
}

export type ImageProviderName = "commons" | "wikipedia-infobox" | "google-cse";

export interface ImageCandidate {
  url: string;
  provider: ImageProviderName;
  /** Sıralama (image-ranker.ts) için aranabilir metin: başlık + açıklama + kategoriler. */
  sourceText: string;
}

export interface RankedImageCandidate extends ImageCandidate {
  score: number;
}
