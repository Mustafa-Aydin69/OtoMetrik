/**
 * Fiyat tahmin formunun paylaşılan tipleri, seçenek listeleri ve doğrulama
 * kuralları. Hem client (PredictionForm) hem server (/api/predict) tarafından
 * kullanılır — bu dosyada "use client" OLMAMALI.
 *
 * Kategori seçenekleri (marka/vites/yakıt/kasa/renk) elle YAZILMAZ —
 * category-options.generated.ts'ten (kaynağı: WebScrape/ai-model/
 * category_mapping.py, modelin gerçek eğitim-zamanı kategorileriyle
 * doğrulanır) türetilir. Kullanıcı dropdown'da Türkçe etiketi görür
 * (örn. "Manuel"); /api/predict'in PREDICTION_API_URL'e ilettiği istekte
 * ise toCanonicalPayload() ile modelin kanonik değeri (örn. "Düz")
 * gönderilir — bkz. dağıtıma hazırlık analizindeki "Manuel/Düz",
 * "Mercedes-Benz/Mercedes - Benz" bulguları.
 */
import {
  KASA_TURU_OPTIONS,
  MARKA_OPTIONS,
  RENK_OPTIONS,
  VITES_OPTIONS,
  YAKIT_TURU_OPTIONS,
  type CategoryOption,
} from "./category-options.generated";
import {
  ENGINE_DISPLACEMENT_MAX,
  ENGINE_POWER_MAX,
  ENGINE_POWER_MIN,
  MAX_TOTAL_DAMAGED_PARTS,
  MILEAGE_MAX,
  PAINTED_PARTS_MAX,
  REPLACED_PARTS_MAX,
  YEAR_MAX,
  YEAR_MIN,
} from "./domain-bounds.generated";
import {
  BODY_TYPE_BY_MODEL,
  ENGINES_BY_MODEL,
  HP_BY_ENGINE,
  MODELS_BY_BRAND,
  PAKET_BY_ENGINE,
  VITES_BY_MODEL,
  type EngineOption,
} from "./vehicle-options.generated";

export interface PredictionInput {
  brand: string;
  model: string;
  year: number;
  mileage: number;
  fuelType: string;
  transmission: string;
  bodyType: string;
  color: string;
  engineDisplacement: number;
  enginePower: number;
  trim: string;
  replacedPartsCount: number;
  paintedPartsCount: number;
  heavyDamage: boolean;
}

export const CURRENT_YEAR = new Date().getFullYear();
// YEAR_MIN/YEAR_MAX (domain-bounds.generated.ts) backend'in kabul ettiği GERÇEK
// aralıktır - MIN_YEAR bunun üzerinden türetilir, ayrıca elle sabit tutulmaz.
export const MIN_YEAR = YEAR_MIN;

// Dropdown'da gösterilen etiketler category-options.generated.ts'ten türetilir
// (elle yazılan bağımsız bir liste YOKTUR) — SelectField zaten string[] beklediği
// için bileşen tarafında değişiklik gerekmez.
const labelsOf = (options: readonly CategoryOption[]): readonly string[] =>
  options.map((o) => o.label);

export const FUEL_TYPES = labelsOf(YAKIT_TURU_OPTIONS);
export const TRANSMISSIONS = labelsOf(VITES_OPTIONS);
export const BODY_TYPES = labelsOf(KASA_TURU_OPTIONS);
export const BRANDS = labelsOf(MARKA_OPTIONS);
export const COLORS = labelsOf(RENK_OPTIONS);

export type FieldErrors = Partial<Record<keyof PredictionInput, string>>;

function isNonNegativeInt(n: number): boolean {
  return Number.isInteger(n) && n >= 0;
}

/** Tamamlanmış (tipli) bir girdiyi doğrular. Hata yoksa boş obje döner. */
export function validatePrediction(input: PredictionInput): FieldErrors {
  const errors: FieldErrors = {};

  if (!input.brand.trim()) errors.brand = "Marka seçin.";
  if (!input.model.trim()) errors.model = "Model girin.";

  if (!Number.isInteger(input.year) || input.year < YEAR_MIN || input.year > YEAR_MAX) {
    errors.year = `Yıl ${YEAR_MIN}–${YEAR_MAX} aralığında olmalı.`;
  }

  if (!Number.isFinite(input.mileage) || input.mileage < 0) {
    errors.mileage = "Kilometre 0 veya daha büyük olmalı.";
  } else if (input.mileage > MILEAGE_MAX) {
    errors.mileage = `Kilometre en fazla ${MILEAGE_MAX.toLocaleString("tr-TR")} olmalı.`;
  }

  if (!(FUEL_TYPES as readonly string[]).includes(input.fuelType)) {
    errors.fuelType = "Yakıt türü seçin.";
  }
  if (!(TRANSMISSIONS as readonly string[]).includes(input.transmission)) {
    errors.transmission = "Vites türü seçin.";
  }
  if (!(BODY_TYPES as readonly string[]).includes(input.bodyType)) {
    errors.bodyType = "Kasa tipi seçin.";
  }
  if (!input.color.trim()) errors.color = "Renk seçin.";

  const isElectric = input.fuelType === "Elektrik";
  if (!isElectric && (!Number.isFinite(input.engineDisplacement) || input.engineDisplacement <= 0)) {
    errors.engineDisplacement = "Motor hacmini cc olarak girin (örn. 1600).";
  } else if (input.engineDisplacement > ENGINE_DISPLACEMENT_MAX) {
    errors.engineDisplacement = `Motor hacmi en fazla ${ENGINE_DISPLACEMENT_MAX} cc olmalı.`;
  }

  if (!Number.isFinite(input.enginePower) || input.enginePower < ENGINE_POWER_MIN) {
    errors.enginePower = "Motor gücünü HP olarak girin (örn. 110).";
  } else if (input.enginePower > ENGINE_POWER_MAX) {
    errors.enginePower = `Motor gücü en fazla ${ENGINE_POWER_MAX} HP olmalı.`;
  }

  if (!isNonNegativeInt(input.replacedPartsCount) || input.replacedPartsCount > REPLACED_PARTS_MAX) {
    errors.replacedPartsCount = `0–${REPLACED_PARTS_MAX} arası tam sayı girin.`;
  }
  if (!isNonNegativeInt(input.paintedPartsCount) || input.paintedPartsCount > PAINTED_PARTS_MAX) {
    errors.paintedPartsCount = `0–${PAINTED_PARTS_MAX} arası tam sayı girin.`;
  }

  // Fiziksel tutarlılık: değişen+boyalı parça toplamı bir aracın sahip
  // olabileceği azami parça sayısını (13) aşamaz - eğitim verisinde bu
  // toplam hiçbir zaman aşılmıyor (bkz. domain_validation.py).
  if (
    isNonNegativeInt(input.replacedPartsCount) &&
    isNonNegativeInt(input.paintedPartsCount) &&
    input.replacedPartsCount + input.paintedPartsCount > MAX_TOTAL_DAMAGED_PARTS
  ) {
    errors.paintedPartsCount = `Değişen ve boyalı parça toplamı en fazla ${MAX_TOTAL_DAMAGED_PARTS} olabilir.`;
  }

  return errors;
}

/**
 * Bilinmeyen JSON gövdesini güvenle PredictionInput'a çevirir (API tarafı).
 * Doğrulama geçmezse input null, errors dolu döner.
 */
export function coercePrediction(
  body: unknown
): { input: PredictionInput | null; errors: FieldErrors | null } {
  if (typeof body !== "object" || body === null) {
    return { input: null, errors: { brand: "Geçersiz istek gövdesi." } };
  }
  const b = body as Record<string, unknown>;

  const str = (v: unknown) => (typeof v === "string" ? v : "");
  const num = (v: unknown) => {
    const n = typeof v === "string" ? Number(v) : typeof v === "number" ? v : NaN;
    return Number.isFinite(n) ? n : NaN;
  };

  const input: PredictionInput = {
    brand: str(b.brand),
    model: str(b.model),
    year: num(b.year),
    mileage: num(b.mileage),
    fuelType: str(b.fuelType),
    transmission: str(b.transmission),
    bodyType: str(b.bodyType),
    color: str(b.color),
    engineDisplacement: num(b.engineDisplacement),
    enginePower: num(b.enginePower),
    trim: str(b.trim),
    replacedPartsCount: num(b.replacedPartsCount),
    paintedPartsCount: num(b.paintedPartsCount),
    heavyDamage: b.heavyDamage === true || b.heavyDamage === "true",
  };

  // Elektrikli araçta motor hacmi girilmemişse 0 kabul edilir.
  if (input.fuelType === "Elektrik" && !Number.isFinite(input.engineDisplacement)) {
    input.engineDisplacement = 0;
  }

  const errors = validatePrediction(input);
  return Object.keys(errors).length > 0
    ? { input: null, errors }
    : { input, errors: null };
}

// PredictionInput alan adı -> o alanın {label,value} seçenek listesi. label->value
// çözümlemesi (toCanonicalPayload) bunun üzerinden yapılır - tek kaynak
// category-options.generated.ts, burada AYRICA bir eşleme tablosu yazılmaz.
const CATEGORICAL_FIELD_OPTIONS: Record<
  "brand" | "transmission" | "fuelType" | "bodyType" | "color",
  readonly CategoryOption[]
> = {
  brand: MARKA_OPTIONS,
  transmission: VITES_OPTIONS,
  fuelType: YAKIT_TURU_OPTIONS,
  bodyType: KASA_TURU_OPTIONS,
  color: RENK_OPTIONS,
};

export function labelToCanonical(field: keyof typeof CATEGORICAL_FIELD_OPTIONS, label: string): string {
  const match = CATEGORICAL_FIELD_OPTIONS[field].find((o) => o.label === label);
  // Eşleşme yoksa (teorik olarak olmamalı - dropdown zaten sadece bilinen
  // etiketleri sunar) etiketi olduğu gibi geçirir; serve.py resolve_canonical()
  // bunu kendi kategori kümesine göre yine de doğrular/reddeder (savunma katmanı).
  return match ? match.value : label;
}

/**
 * PredictionInput'u (kullanıcı etiketleriyle) modelin beklediği kanonik
 * kategori değerlerine çevirir — /api/predict, PREDICTION_API_URL'e bu
 * çevrilmiş halini gönderir. Diğer (kategorik olmayan) alanlar değişmeden
 * kopyalanır.
 */
export function toCanonicalPayload(input: PredictionInput): PredictionInput {
  return {
    ...input,
    brand: labelToCanonical("brand", input.brand),
    transmission: labelToCanonical("transmission", input.transmission),
    fuelType: labelToCanonical("fuelType", input.fuelType),
    bodyType: labelToCanonical("bodyType", input.bodyType),
    color: labelToCanonical("color", input.color),
  };
}

/**
 * labelToCanonical'ın tersi: modelin kanonik değerini (örn. "LPG & Benzin",
 * Motor adımından gelir) kullanıcıya gösterilecek Türkçe etikete (örn. "LPG")
 * çevirir. Eşleşme yoksa (teorik olarak olmamalı - kanonik değer zaten
 * CATEGORICAL_FIELD_OPTIONS'tan geliyor) kanonik değeri olduğu gibi döner.
 */
export function canonicalToLabel(field: keyof typeof CATEGORICAL_FIELD_OPTIONS, canonical: string): string {
  const match = CATEGORICAL_FIELD_OPTIONS[field].find((o) => o.value === canonical);
  return match ? match.label : canonical;
}

const trCollator = new Intl.Collator("tr", { numeric: true, sensitivity: "base" });

export type { EngineOption };

export function vehicleKey(canonicalBrand: string, model: string): string {
  return `${canonicalBrand}|${model.trim()}`;
}

function engineKey(canonicalBrand: string, model: string, hacmiBucket: number, yakitTuru: string): string {
  return `${vehicleKey(canonicalBrand, model)}|${hacmiBucket.toFixed(1)}|${yakitTuru}`;
}

/**
 * Seçilen marka (website etiketi) için eğitim verisinde GERÇEKTEN görülen
 * modelleri tr-collator sırasına göre döner — VehicleSelector'ın ikinci
 * kademesi. Bilinmeyen/veri dışı marka için boş dizi döner.
 */
export function getModelsForBrand(brandLabel: string): readonly string[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  const models = MODELS_BY_BRAND[canonicalBrand];
  return models ? [...models].sort(trCollator.compare) : [];
}

/**
 * Seçilen marka+model için eğitim verisinde görülen motor (hacim kovası +
 * yakıt türü) kombinasyonlarını hacme göre artan sırada döner —
 * VehicleSelector'ın üçüncü kademesi (Motor). Her seçeneğin `exactCc` alanı
 * o kovadaki EN SIK GÖRÜLEN gerçek motor_hacmi değeridir; /predict'e
 * GÖNDERİLMESİ gereken budur, `hacmiBucket`'ın kendisi değil (bkz.
 * generate_vehicle_options.py modül notu).
 */
export function getEnginesForModel(brandLabel: string, model: string): readonly EngineOption[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  return ENGINES_BY_MODEL[vehicleKey(canonicalBrand, model)] ?? [];
}

/**
 * Seçilen marka+model+motor için eğitimde GERÇEKTEN görülen paket
 * değerlerini (en sıktan en nadire) döner — VehicleSelector'ın dördüncü
 * (opsiyonel) kademesi. Boş dizi, panelin paket adımını atlamasına izin
 * verir. Kullanıcı önerilerden birini seçerse gönderilen değer zaten
 * kanoniktir (paket için ayrı bir label→canonical çevirisi YOKTUR).
 */
export function getPaketOptions(
  brandLabel: string,
  model: string,
  hacmiBucket: number,
  yakitTuru: string
): readonly string[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  return PAKET_BY_ENGINE[engineKey(canonicalBrand, model, hacmiBucket, yakitTuru)] ?? [];
}

/**
 * Seçilen marka+model+motor için eğitimde görülen ayrık motor gücü (HP)
 * değerlerini artan sırada döner. PredictionForm bu listeyi kullanarak: 1
 * değer varsa alanı otomatik doldurup kilitler, >1 değer varsa seçilebilir
 * bir dropdown gösterir, 0 değer varsa (nadiren) serbest girişe düşer.
 */
export function getHpOptions(
  brandLabel: string,
  model: string,
  hacmiBucket: number,
  yakitTuru: string
): readonly number[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  return HP_BY_ENGINE[engineKey(canonicalBrand, model, hacmiBucket, yakitTuru)] ?? [];
}

/**
 * Seçilen marka+model için eğitimde GERÇEKTEN görülen kasa tiplerini (en
 * sıktan en nadire) etiket olarak döner — örn. Citroën Berlingo için
 * ["Camlı Van", "Panel Van", ...], "Sedan"/"MPV" gibi bu araca hiç uymayan
 * seçenekler döndürmez. PredictionForm bu listeyi kullanarak: 1 değer varsa
 * alanı otomatik doldurup kilitler, >1 değer varsa yalnızca bu değerlerden
 * oluşan bir dropdown gösterir, 0 değer varsa (araç henüz seçilmedi ya da bu
 * marka+model için veri yok) tam kanonik listeye (BODY_TYPES) serbest
 * seçime düşer.
 */
export function getBodyTypesForModel(brandLabel: string, model: string): readonly string[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  const values = BODY_TYPE_BY_MODEL[vehicleKey(canonicalBrand, model)] ?? [];
  return values.map((v) => canonicalToLabel("bodyType", v));
}

/**
 * getBodyTypesForModel ile ayni desen (bkz. o fonksiyonun docstring'i) -
 * marka+model icin egitimde GERCEKTEN gorulen vites turlerini doner.
 */
export function getVitesOptionsForModel(brandLabel: string, model: string): readonly string[] {
  const canonicalBrand = labelToCanonical("brand", brandLabel);
  const values = VITES_BY_MODEL[vehicleKey(canonicalBrand, model)] ?? [];
  return values.map((v) => canonicalToLabel("transmission", v));
}
