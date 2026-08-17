/**
 * Araç görsel bulma pipeline'ının katmanlar arası paylaşılan tipleri. Akış:
 *
 *   Raw Vehicle Input
 *     -> vehicle-parser.ts        (Vehicle Normalization)
 *     -> vehicle-aliases.ts       (Vehicle Identity Resolution - marka/model kanonikleştirme)
 *     -> generation-resolver.ts   (Generation / Facelift Resolution)
 *     -> query-generator.ts       (Search Query Generation)
 *     -> providers.ts             (Candidate Retrieval)
 *     -> candidate-ranker.ts      (Hard Validation/Conflict Filtering + Metadata Scoring)
 *     -> confidence.ts            (Confidence Gate)
 *     -> vehicle-image.service.ts (orkestrasyon, Image Result)
 *
 * Bkz. her modülün kendi docstring'i için detay.
 */

/** Ham form girdisinden türetilmiş, pipeline boyunca taşınan tek canonical araç kimliği. */
export interface VehicleIdentity {
  make: string;
  /** İngilizceye çevrilmiş/kanonikleştirilmiş model adı (bkz. vehicle-aliases.ts) - arama sorgularında kullanılan budur. */
  model: string;
  /** Ham (form'dan gelen, çevrilmemiş) model adı - loglama/karşılaştırma için. */
  rawModel: string;
  /** paket/trim metninden ayrıştırılabildiyse motor/alt-model kodu (örn. "320i") - bkz. vehicle-parser.ts extractVariant(). */
  variant: string | null;
  /** variant çıkarıldıktan sonra kalan trim metni (örn. "Executive M Sport") - ayrıştırılamadıysa ham trim'in TAMAMI burada kalır. */
  trim: string | null;
  year: number | null;
  bodyType: string | null;
  /** İngilizce (arama motorları/Commons için) - vehicle-parser.ts çevirir. */
  color: string | null;
  /** Nesil kodu (örn. "G20") - generation-resolver.ts doldurur, bilinmiyorsa null. */
  generation: string | null;
  /** Wikipedia'nın sıra sayısı ifadesi (örn. "Seventh generation") - ikincil arama anahtar kelimesi. */
  generationOrdinalLabel: string | null;
  /** Facelift kodu (örn. "LCI") - yalnızca KESİN biliniyorsa doldurulur, aksi halde null (bkz. generation-rules.ts). */
  facelift: string | null;
  generationStartYear: number | null;
  generationEndYear: number | null;
  /** Nesil hangi katmandan geldi - debug/gözlemlenebilirlik için. */
  generationSource: "local" | "wikipedia" | "unknown";
}

/** generation-rules.ts'teki yerel/deterministik nesil bilgisi - Wikipedia'ya gitmeden önce denenir. */
export interface GenerationRule {
  make: string;
  model: string;
  generation: string;
  ordinalLabel: string;
  startYear: number;
  /** Tanımsızsa bu hâlâ güncel/son nesil demektir. */
  endYear?: number;
  facelift?: string;
  faceliftStartYear?: number;
}

export interface ResolvedGeneration {
  label: string;
  ordinalLabel: string;
  allCodes: string[];
  startYear: number;
  endYear: number | null;
  facelift: string | null;
  source: "local" | "wikipedia";
}

export type ImageProviderName = "commons" | "wikipedia-infobox" | "google-cse";

/** Bir sağlayıcının döndürdüğü, henüz değerlendirilmemiş ham aday - candidate-ranker.ts bunu işler. */
export interface ImageCandidate {
  url: string;
  pageUrl?: string;
  filename?: string;
  title?: string;
  description?: string;
  categories?: string[];
  source: ImageProviderName;
}

/** Hangi kimlik alanlarının adayın metadata'sında doğrulandığı - confidence.ts ve debug log için. */
export interface MatchedFields {
  make: boolean;
  model: boolean;
  generation: boolean;
  /** Yıl hiç doğrulanamadıysa (ne uyumlu ne çelişkili) "unknown" - bkz. candidate-ranker.ts isYearCompatible. */
  year: boolean | "unknown";
  variant: boolean;
  trim: boolean;
  bodyType: boolean;
  color: boolean;
}

export interface RankedImageCandidate extends ImageCandidate {
  score: number;
  matched: MatchedFields;
  /** true ise hard-reject uygulandı (candidate-ranker.ts) - confidence.ts bunu asla döndürmez. */
  rejected: boolean;
  rejectionReason?: string;
}

export interface ImageSelectionResult {
  imageUrl: string | null;
  /** 0-1 arası - bkz. confidence.ts eşik mantığı. */
  confidence: number;
  candidateScore?: number;
  matched?: MatchedFields;
  rejectedCount: number;
  source?: ImageProviderName;
}

/**
 * Görsel kaynağı sağlayıcı arayüzü - şu an tek implementasyon Commons
 * (+ opsiyonel Google CSE, + son çare Wikipedia infobox), bkz. providers.ts.
 * İleride yeni bir kaynak (örn. üretici basın görselleri) bu arayüzü
 * uygulayıp providers dizisine eklenebilir - vehicle-image.service.ts
 * değişmeden kalır.
 */
export interface VehicleImageProvider {
  name: ImageProviderName;
  search(query: string, limit: number): Promise<ImageCandidate[]>;
}
