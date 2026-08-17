/**
 * vehicle-image-audit.ts ve query-quality-audit.ts'in PAYLAŞTIĞI test
 * matrisi - üretim koduna dokunmaz. Bkz. vehicle-image-audit.ts docstring'i.
 */
import type { VehicleFormInput } from "../src/lib/vehicle-image/vehicle-parser";

export interface TestCase extends VehicleFormInput {
  id: string;
  category: string;
}

// Test matrisi: dataset-frequency.ts çıktısındaki GERÇEK en sık marka/modellerden
// (bkz. scripts/audit-results/dataset-frequency.json) + kullanıcının istediği
// spesifik edge-case'ler (BMW N Serisi, Mercedes bare kod, Audi nesil kodları,
// VW Golf Mk5-8, Renault Clio III/IV/V, Megane II/III/IV, Peugeot 208/308 nesil
// geçişleri). Model adları dataset'teki RAW (Türkçe) haliyle - VehicleSelector'ın
// gerçekte gönderdiği değerler bunlar.
export const CASES: TestCase[] = [
  // --- BMW: N Serisi edge case + eski/orta/yeni nesil ---
  { id: "bmw-3-e46", category: "BMW-edge", brand: "BMW", model: "3 Serisi", year: 2001, trim: "320i", bodyType: "Sedan", color: "Gri" },
  { id: "bmw-3-f30", category: "BMW-edge", brand: "BMW", model: "3 Serisi", year: 2012, trim: "320d", bodyType: "Sedan", color: "Siyah" },
  { id: "bmw-3-g20", category: "BMW-edge", brand: "BMW", model: "3 Serisi", year: 2023, trim: "320i Executive M Sport", bodyType: "Sedan", color: "Beyaz" },
  { id: "bmw-5-e39", category: "BMW-edge", brand: "BMW", model: "5 Serisi", year: 1999, trim: "523i", bodyType: "Sedan", color: "Gri" },
  { id: "bmw-5-g30", category: "BMW-edge", brand: "BMW", model: "5 Serisi", year: 2018, trim: "520d", bodyType: "Sedan", color: "Siyah" },
  { id: "bmw-1-f20", category: "BMW-edge", brand: "BMW", model: "1 Serisi", year: 2015, trim: "116i", bodyType: "Hatchback", color: "Kırmızı" },

  // --- Mercedes: bare-letter model kodu edge case ---
  { id: "mb-c-w202", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "C", year: 1998, trim: "C 180", bodyType: "Sedan", color: "Gümüş" },
  { id: "mb-c-w204", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "C", year: 2010, trim: "C 200", bodyType: "Sedan", color: "Siyah" },
  { id: "mb-c-w206", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "C", year: 2022, trim: "C 220d", bodyType: "Sedan", color: "Beyaz" },
  { id: "mb-e-w211", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "E", year: 2005, trim: "E 220", bodyType: "Sedan", color: "Gri" },
  { id: "mb-e-w213", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "E", year: 2020, trim: "E 200", bodyType: "Sedan", color: "Siyah" },
  { id: "mb-a-w177", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "A", year: 2019, trim: "A 200", bodyType: "Hatchback", color: "Beyaz" },
  { id: "mb-cla", category: "Mercedes-edge", brand: "Mercedes-Benz", model: "CLA", year: 2015, trim: "CLA 200", bodyType: "Coupe", color: "Kırmızı" },

  // --- Audi: nesil kodu (8P/8V/8Y, B8/B9, C6/C7) edge case ---
  { id: "audi-a3-8p", category: "Audi-edge", brand: "Audi", model: "A3", year: 2005, trim: "1.6", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "audi-a3-8v", category: "Audi-edge", brand: "Audi", model: "A3", year: 2018, trim: "1.6 TDI", bodyType: "Sedan", color: "Beyaz" },
  { id: "audi-a4-b8", category: "Audi-edge", brand: "Audi", model: "A4", year: 2010, trim: "2.0 TDI", bodyType: "Sedan", color: "Siyah" },
  { id: "audi-a4-b9", category: "Audi-edge", brand: "Audi", model: "A4", year: 2020, trim: "2.0 TDI", bodyType: "Sedan", color: "Gri" },
  { id: "audi-a6-c6", category: "Audi-edge", brand: "Audi", model: "A6", year: 2005, trim: "2.4", bodyType: "Sedan", color: "Gümüş" },

  // --- Volkswagen: Golf Mk5-8 nesil ayrımı ---
  { id: "vw-golf-mk5", category: "VW-golf", brand: "Volkswagen", model: "Golf", year: 2005, trim: "1.6", bodyType: "Hatchback (5 Kapı)", color: "Mavi" },
  { id: "vw-golf-mk6", category: "VW-golf", brand: "Volkswagen", model: "Golf", year: 2010, trim: "1.6 TDI", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "vw-golf-mk7", category: "VW-golf", brand: "Volkswagen", model: "Golf", year: 2015, trim: "1.6 TDI Comfortline", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "vw-golf-mk8", category: "VW-golf", brand: "Volkswagen", model: "Golf", year: 2021, trim: "1.5 TSI Life", bodyType: "Hatchback (5 Kapı)", color: "Siyah" },
  { id: "vw-passat-b5", category: "VW-other", brand: "Volkswagen", model: "Passat", year: 2000, trim: "1.8", bodyType: "Sedan", color: "Gri" },
  { id: "vw-passat-b8", category: "VW-other", brand: "Volkswagen", model: "Passat", year: 2018, trim: "1.6 TDI", bodyType: "Sedan", color: "Beyaz" },
  { id: "vw-polo", category: "VW-other", brand: "Volkswagen", model: "Polo", year: 2012, trim: "1.6 TDI", bodyType: "Hatchback (5 Kapı)", color: "Kırmızı" },
  { id: "vw-tiguan", category: "VW-other", brand: "Volkswagen", model: "Tiguan", year: 2020, trim: "1.5 TSI", bodyType: "SUV", color: "Beyaz" },

  // --- Toyota ---
  { id: "toyota-corolla-old", category: "Toyota", brand: "Toyota", model: "Corolla", year: 2008, trim: "1.4 D-4D", bodyType: "Sedan", color: "Gümüş" },
  { id: "toyota-corolla-new", category: "Toyota", brand: "Toyota", model: "Corolla", year: 2020, trim: "1.6", bodyType: "Sedan", color: "Beyaz" },
  { id: "toyota-yaris", category: "Toyota", brand: "Toyota", model: "Yaris", year: 2015, trim: "1.4 D-4D", bodyType: "Hatchback (5 Kapı)", color: "Kırmızı" },
  { id: "toyota-chr", category: "Toyota", brand: "Toyota", model: "C-HR", year: 2019, trim: "1.2", bodyType: "SUV", color: "Turuncu" },

  // --- Renault: Clio III/IV/V ve Megane II/III/IV ayrımı ---
  { id: "renault-clio-3", category: "Renault-clio", brand: "Renault", model: "Clio", year: 2010, trim: "1.5 dCi", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "renault-clio-4", category: "Renault-clio", brand: "Renault", model: "Clio", year: 2016, trim: "1.5 dCi Touch", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "renault-clio-5", category: "Renault-clio", brand: "Renault", model: "Clio", year: 2021, trim: "1.0 TCe", bodyType: "Hatchback (5 Kapı)", color: "Kırmızı" },
  { id: "renault-megane-2", category: "Renault-megane", brand: "Renault", model: "Megane", year: 2005, trim: "1.5 dCi", bodyType: "Sedan", color: "Gri" },
  { id: "renault-megane-3", category: "Renault-megane", brand: "Renault", model: "Megane", year: 2012, trim: "1.5 dCi", bodyType: "Hatchback (5 Kapı)", color: "Siyah" },
  { id: "renault-megane-4", category: "Renault-megane", brand: "Renault", model: "Megane", year: 2019, trim: "1.3 TCe", bodyType: "Sedan", color: "Beyaz" },
  { id: "renault-talisman", category: "Renault-other", brand: "Renault", model: "Talisman", year: 2018, trim: "1.5 dCi", bodyType: "Sedan", color: "Gri" },
  { id: "renault-captur", category: "Renault-other", brand: "Renault", model: "Captur", year: 2020, trim: "1.5 dCi", bodyType: "SUV", color: "Turuncu" },

  // --- Peugeot: 208/308 tam yeniden tasarım nesil geçişleri ---
  { id: "peugeot-208-mk1", category: "Peugeot-208", brand: "Peugeot", model: "208", year: 2014, trim: "1.4 HDi", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "peugeot-208-mk2", category: "Peugeot-208", brand: "Peugeot", model: "208", year: 2021, trim: "1.2 PureTech", bodyType: "Hatchback (5 Kapı)", color: "Mavi" },
  { id: "peugeot-308-mk1", category: "Peugeot-308", brand: "Peugeot", model: "308", year: 2010, trim: "1.6 HDi", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "peugeot-308-mk2", category: "Peugeot-308", brand: "Peugeot", model: "308", year: 2015, trim: "1.6 BlueHDi", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "peugeot-3008", category: "Peugeot-other", brand: "Peugeot", model: "3008", year: 2018, trim: "1.6 BlueHDi", bodyType: "SUV", color: "Siyah" },
  { id: "peugeot-508", category: "Peugeot-other", brand: "Peugeot", model: "508", year: 2019, trim: "1.5 BlueHDi", bodyType: "Sedan", color: "Gri" },

  // --- Citroen ---
  { id: "citroen-c3", category: "Citroen", brand: "Citroen", model: "C3", year: 2015, trim: "1.4 HDi", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "citroen-c4", category: "Citroen", brand: "Citroen", model: "C4", year: 2012, trim: "1.6 HDi", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "citroen-berlingo", category: "Citroen", brand: "Citroen", model: "Berlingo", year: 2018, trim: "1.6 BlueHDi", bodyType: "Panelvan", color: "Beyaz" },

  // --- Ford ---
  { id: "ford-focus-mk2", category: "Ford", brand: "Ford", model: "Focus", year: 2005, trim: "1.6 TDCi", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "ford-focus-mk4", category: "Ford", brand: "Ford", model: "Focus", year: 2018, trim: "1.5 TDCi", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "ford-fiesta", category: "Ford", brand: "Ford", model: "Fiesta", year: 2015, trim: "1.5 TDCi", bodyType: "Hatchback (5 Kapı)", color: "Kırmızı" },

  // --- Fiat ---
  { id: "fiat-egea", category: "Fiat", brand: "Fiat", model: "Egea", year: 2018, trim: "1.6 Multijet", bodyType: "Sedan", color: "Beyaz" },
  { id: "fiat-linea", category: "Fiat", brand: "Fiat", model: "Linea", year: 2012, trim: "1.3 Multijet", bodyType: "Sedan", color: "Gri" },
  { id: "fiat-doblo", category: "Fiat", brand: "Fiat", model: "Doblo", year: 2016, trim: "1.6 Multijet", bodyType: "Panelvan", color: "Beyaz" },

  // --- Honda / Hyundai / Kia / Skoda / Seat / Volvo / Opel ---
  { id: "honda-civic-old", category: "Honda", brand: "Honda", model: "Civic", year: 2010, trim: "1.6", bodyType: "Sedan", color: "Gri" },
  { id: "honda-civic-new", category: "Honda", brand: "Honda", model: "Civic", year: 2020, trim: "1.5 VTEC", bodyType: "Sedan", color: "Beyaz" },
  { id: "hyundai-i20", category: "Hyundai", brand: "Hyundai", model: "i20", year: 2017, trim: "1.4", bodyType: "Hatchback (5 Kapı)", color: "Kırmızı" },
  { id: "hyundai-tucson", category: "Hyundai", brand: "Hyundai", model: "Tucson", year: 2020, trim: "1.6 CRDi", bodyType: "SUV", color: "Beyaz" },
  { id: "kia-sportage", category: "Kia", brand: "Kia", model: "Sportage", year: 2019, trim: "1.6 CRDi", bodyType: "SUV", color: "Gri" },
  { id: "kia-rio", category: "Kia", brand: "Kia", model: "Rio", year: 2015, trim: "1.4", bodyType: "Sedan", color: "Beyaz" },
  { id: "skoda-octavia", category: "Skoda", brand: "Skoda", model: "Octavia", year: 2015, trim: "1.6 TDI", bodyType: "Sedan", color: "Gri" },
  { id: "seat-leon", category: "Seat", brand: "Seat", model: "Leon", year: 2014, trim: "1.6 TDI", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
  { id: "volvo-s60", category: "Volvo", brand: "Volvo", model: "S60", year: 2016, trim: "D3", bodyType: "Sedan", color: "Siyah" },
  { id: "opel-astra", category: "Opel", brand: "Opel", model: "Astra", year: 2012, trim: "1.6 CDTI", bodyType: "Hatchback (5 Kapı)", color: "Gri" },
  { id: "opel-corsa", category: "Opel", brand: "Opel", model: "Corsa", year: 2018, trim: "1.4", bodyType: "Hatchback (5 Kapı)", color: "Beyaz" },
];
