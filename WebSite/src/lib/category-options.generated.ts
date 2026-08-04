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
  { label: "Alfa Romeo", value: "Alfa Romeo" },
  { label: "Aston Martin", value: "Aston Martin" },
  { label: "Audi", value: "Audi" },
  { label: "Bentley", value: "Bentley" },
  { label: "BMW", value: "BMW" },
  { label: "Buick", value: "Buick" },
  { label: "BYD", value: "BYD" },
  { label: "Cadillac", value: "Cadillac" },
  { label: "Chery", value: "Chery" },
  { label: "Chevrolet", value: "Chevrolet" },
  { label: "Chrysler", value: "Chrysler" },
  { label: "Citroën", value: "Citroen" },
  { label: "Cupra", value: "Cupra" },
  { label: "Dacia", value: "Dacia" },
  { label: "Daewoo", value: "Daewoo" },
  { label: "Daihatsu", value: "Daihatsu" },
  { label: "Dodge", value: "Dodge" },
  { label: "DS Automobiles", value: "DS Automobiles" },
  { label: "Fiat", value: "Fiat" },
  { label: "Ford", value: "Ford" },
  { label: "Geely", value: "Geely" },
  { label: "Honda", value: "Honda" },
  { label: "Hyundai", value: "Hyundai" },
  { label: "Ikco", value: "Ikco" },
  { label: "Infiniti", value: "Infiniti" },
  { label: "Isuzu", value: "Isuzu" },
  { label: "Iveco-Otoyol", value: "Iveco - Otoyol" },
  { label: "Jaecoo", value: "Jaecoo" },
  { label: "Jaguar", value: "Jaguar" },
  { label: "Jeep", value: "Jeep" },
  { label: "Kia", value: "Kia" },
  { label: "Lada", value: "Lada" },
  { label: "Lancia", value: "Lancia" },
  { label: "Land Rover", value: "Land Rover" },
  { label: "Lexus", value: "Lexus" },
  { label: "Lincoln", value: "Lincoln" },
  { label: "Lotus", value: "Lotus" },
  { label: "Maserati", value: "Maserati" },
  { label: "Maxus", value: "Maxus" },
  { label: "Mazda", value: "Mazda" },
  { label: "Mercedes-Benz", value: "Mercedes - Benz" },
  { label: "Mercury", value: "Mercury" },
  { label: "MG", value: "MG" },
  { label: "Mini", value: "Mini" },
  { label: "Mitsubishi", value: "Mitsubishi" },
  { label: "Nissan", value: "Nissan" },
  { label: "Opel", value: "Opel" },
  { label: "Peugeot", value: "Peugeot" },
  { label: "Pontiac", value: "Pontiac" },
  { label: "Porsche", value: "Porsche" },
  { label: "Proton", value: "Proton" },
  { label: "Renault", value: "Renault" },
  { label: "Rolls-Royce", value: "Rolls-Royce" },
  { label: "Rover", value: "Rover" },
  { label: "Saab", value: "Saab" },
  { label: "Seat", value: "Seat" },
  { label: "Skoda", value: "Skoda" },
  { label: "Smart", value: "Smart" },
  { label: "Ssangyong", value: "Ssangyong" },
  { label: "Subaru", value: "Subaru" },
  { label: "Suzuki", value: "Suzuki" },
  { label: "Tata", value: "Tata" },
  { label: "Tesla", value: "Tesla" },
  { label: "Tofaş", value: "Tofaş" },
  { label: "TOGG", value: "TOGG" },
  { label: "Toyota", value: "Toyota" },
  { label: "Volkswagen", value: "Volkswagen" },
  { label: "Volvo", value: "Volvo" },
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
