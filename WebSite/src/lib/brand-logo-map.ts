/**
 * Kanonik marka değeri (category-options.generated.ts MARKA_OPTIONS[].value
 * ile birebir aynı) -> simple-icons SVG dosya adı (uzantısız), WebSite/public/
 * brand-logos/{slug}.svg altında. `null` = bu markanın açık lisanslı bir
 * logosu bulunamadı, BrandCard baş harf rozetine düşer.
 *
 * ELLE BAKIMLI - marka listesi genişlerse (category_mapping.py) buraya da
 * yeni bir satır eklenmeli. Otomatik üretilmiyor çünkü logo varlığı eğitim
 * verisinden türetilemez (bkz. plan: "kod değil, statik varlık").
 */
export const BRAND_LOGO_SLUG: Record<string, string | null> = {
  "Alfa Romeo": "alfaromeo",
  "Aston Martin": "astonmartin",
  Audi: "audi",
  Bentley: "bentley",
  BMW: "bmw",
  Buick: null,
  BYD: null,
  Cadillac: "cadillac",
  Chery: null,
  Chevrolet: "chevrolet",
  Chrysler: "chrysler",
  Citroen: "citroen",
  Cupra: null,
  Dacia: "dacia",
  Daewoo: null,
  Daihatsu: null,
  Dodge: null,
  "DS Automobiles": "dsautomobiles",
  Fiat: "fiat",
  Ford: "ford",
  Geely: null,
  Honda: "honda",
  Hyundai: "hyundai",
  Ikco: null,
  Infiniti: "infiniti",
  Isuzu: null,
  "Iveco - Otoyol": "iveco",
  Jaecoo: null,
  Jaguar: "jaguar",
  Jeep: "jeep",
  Kia: "kia",
  Lada: "lada",
  Lancia: null,
  "Land Rover": "landrover",
  Lexus: null,
  Lincoln: null,
  Lotus: null,
  Maserati: "maserati",
  Maxus: null,
  Mazda: "mazda",
  "Mercedes - Benz": "mercedes",
  Mercury: null,
  MG: "mg",
  Mini: "mini",
  Mitsubishi: "mitsubishi",
  Nissan: "nissan",
  Opel: "opel",
  Peugeot: "peugeot",
  Pontiac: null,
  Porsche: "porsche",
  Proton: "proton",
  Renault: "renault",
  "Rolls-Royce": "rollsroyce",
  Rover: null,
  Saab: null,
  Seat: "seat",
  Skoda: "skoda",
  Smart: "smart",
  Ssangyong: null,
  Subaru: "subaru",
  Suzuki: "suzuki",
  Tata: "tata",
  Tesla: "tesla",
  Tofaş: null,
  TOGG: null,
  Toyota: "toyota",
  Volkswagen: "volkswagen",
  Volvo: "volvo",
};

export function getBrandLogoSlug(canonicalBrand: string): string | null {
  return BRAND_LOGO_SLUG[canonicalBrand] ?? null;
}
