/**
 * OTOMATIK URETILMISTIR - ELLE DUZENLEMEYIN.
 *
 * Kaynak: WebScrape/ai-model/category_mapping.py (TEK dogruluk kaynagi -
 * model artefaktinin egitim-zamani kategorileriyle dogrulanir).
 * Uretmek icin: cd WebScrape/ai-model && python generate_category_mapping.py
 *
 * label: kullaniciya gosterilen Turkce etiket (dropdown'da gorunen).
 * value: modelin egitim-zamaninda gordugu kanonik kategori degeri
 *        (/predict'e GONDERILMESI gereken deger).
 */

export interface CategoryOption {
  label: string;
  value: string;
}

export const VITES_OPTIONS: CategoryOption[] = [
  { label: "Manuel", value: "Düz" },
  { label: "Otomatik", value: "Otomatik" },
  { label: "Yarı Otomatik", value: "Yarı Otomatik" },
];

export const YAKIT_TURU_OPTIONS: CategoryOption[] = [
  { label: "Benzin", value: "Benzin" },
  { label: "Dizel", value: "Dizel" },
  { label: "LPG", value: "LPG & Benzin" },
  { label: "Hibrit", value: "Hibrit" },
  { label: "Elektrik", value: "Elektrik" },
];

export const MARKA_OPTIONS: CategoryOption[] = [
  { label: "Audi", value: "Audi" },
  { label: "BMW", value: "BMW" },
  { label: "Citroën", value: "Citroen" },
  { label: "Dacia", value: "Dacia" },
  { label: "Fiat", value: "Fiat" },
  { label: "Ford", value: "Ford" },
  { label: "Honda", value: "Honda" },
  { label: "Hyundai", value: "Hyundai" },
  { label: "Kia", value: "Kia" },
  { label: "Mercedes-Benz", value: "Mercedes - Benz" },
  { label: "Nissan", value: "Nissan" },
  { label: "Opel", value: "Opel" },
  { label: "Peugeot", value: "Peugeot" },
  { label: "Renault", value: "Renault" },
  { label: "Seat", value: "Seat" },
  { label: "Skoda", value: "Skoda" },
  { label: "Toyota", value: "Toyota" },
  { label: "Volkswagen", value: "Volkswagen" },
  { label: "Volvo", value: "Volvo" },
  { label: "Diğer", value: "" },
];

export const RENK_OPTIONS: CategoryOption[] = [
  { label: "Siyah", value: "Siyah" },
  { label: "Beyaz", value: "Beyaz" },
  { label: "Gri", value: "Gri" },
  { label: "Gümüş", value: "Gri (Gümüş)" },
  { label: "Mavi", value: "Mavi" },
  { label: "Kırmızı", value: "Kırmızı" },
  { label: "Kahverengi", value: "Kahverengi" },
  { label: "Bej", value: "Bej" },
  { label: "Yeşil", value: "Yeşil" },
  { label: "Turuncu", value: "Turuncu" },
  { label: "Sarı", value: "Sarı" },
  { label: "Bordo", value: "Bordo" },
  { label: "Lacivert", value: "Lacivert" },
  { label: "Diğer", value: "Diğer" },
];

export const KASA_TURU_OPTIONS: CategoryOption[] = [
  { label: "Sedan", value: "Sedan" },
  { label: "Hatchback (3 Kapı)", value: "Hatchback/3" },
  { label: "Hatchback (5 Kapı)", value: "Hatchback/5" },
  { label: "SUV", value: "SUV" },
  { label: "Coupe", value: "Coupe" },
  { label: "Station Wagon", value: "Station wagon" },
  { label: "Cabrio", value: "Cabrio" },
  { label: "Pick-up", value: "Pick-up" },
  { label: "MPV", value: "MPV" },
  { label: "Panelvan", value: "Panel Van" },
  { label: "Crossover", value: "Crossover" },
];
