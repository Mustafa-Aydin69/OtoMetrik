/**
 * Fiyat tahmin formunun paylaşılan tipleri, seçenek listeleri ve doğrulama
 * kuralları. Hem client (PredictionForm) hem server (/api/predict) tarafından
 * kullanılır — bu dosyada "use client" OLMAMALI.
 */

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
export const MIN_YEAR = 1960;

export const FUEL_TYPES = ["Benzin", "Dizel", "LPG", "Hibrit", "Elektrik"] as const;

export const TRANSMISSIONS = ["Manuel", "Otomatik", "Yarı Otomatik"] as const;

export const BODY_TYPES = [
  "Sedan",
  "Hatchback",
  "SUV",
  "Coupe",
  "Station Wagon",
  "Cabrio",
  "Pick-up",
  "MPV",
  "Panelvan",
  "Minivan",
] as const;

export const BRANDS = [
  "Audi",
  "BMW",
  "Citroën",
  "Dacia",
  "Fiat",
  "Ford",
  "Honda",
  "Hyundai",
  "Kia",
  "Mercedes-Benz",
  "Nissan",
  "Opel",
  "Peugeot",
  "Renault",
  "Seat",
  "Skoda",
  "Toyota",
  "Volkswagen",
  "Volvo",
  "Diğer",
] as const;

export const COLORS = [
  "Siyah",
  "Beyaz",
  "Gri",
  "Gümüş",
  "Mavi",
  "Kırmızı",
  "Kahverengi",
  "Bej",
  "Yeşil",
  "Turuncu",
  "Sarı",
  "Bordo",
  "Lacivert",
  "Diğer",
] as const;

export type FieldErrors = Partial<Record<keyof PredictionInput, string>>;

function isNonNegativeInt(n: number): boolean {
  return Number.isInteger(n) && n >= 0;
}

/** Tamamlanmış (tipli) bir girdiyi doğrular. Hata yoksa boş obje döner. */
export function validatePrediction(input: PredictionInput): FieldErrors {
  const errors: FieldErrors = {};

  if (!input.brand.trim()) errors.brand = "Marka seçin.";
  if (!input.model.trim()) errors.model = "Model girin.";

  if (!Number.isInteger(input.year) || input.year < MIN_YEAR || input.year > CURRENT_YEAR + 1) {
    errors.year = `Yıl ${MIN_YEAR}–${CURRENT_YEAR + 1} aralığında olmalı.`;
  }

  if (!Number.isFinite(input.mileage) || input.mileage < 0) {
    errors.mileage = "Kilometre 0 veya daha büyük olmalı.";
  } else if (input.mileage > 2_000_000) {
    errors.mileage = "Kilometre gerçekçi bir değer olmalı.";
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
  } else if (input.engineDisplacement > 9000) {
    errors.engineDisplacement = "Motor hacmi gerçekçi bir değer olmalı.";
  }

  if (!Number.isFinite(input.enginePower) || input.enginePower <= 0) {
    errors.enginePower = "Motor gücünü HP olarak girin (örn. 110).";
  } else if (input.enginePower > 2000) {
    errors.enginePower = "Motor gücü gerçekçi bir değer olmalı.";
  }

  if (!isNonNegativeInt(input.replacedPartsCount) || input.replacedPartsCount > 13) {
    errors.replacedPartsCount = "0–13 arası tam sayı girin.";
  }
  if (!isNonNegativeInt(input.paintedPartsCount) || input.paintedPartsCount > 13) {
    errors.paintedPartsCount = "0–13 arası tam sayı girin.";
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
