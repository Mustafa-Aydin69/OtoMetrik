# 🚀 OtoMetrik: Uçtan Uca Araç Fiyat Tahminleme ve Veri Hattı

**OtoMetrik**, hedef ikinci el araç ilan platformundan yüksek performanslı asenkron web kazıma teknikleriyle güncel piyasa verilerini toplayan, bunu açık (public) bir referans veri setiyle harmanlayan, temizleyen ve makine öğrenmesi modelleri için yapılandırılmış veri setlerine dönüştüren gelişmiş bir büyük veri hattı (Data Pipeline) projesidir.

Sistem; birbirinden tamamen izole edilmiş (decoupled) mikroservis bileşenleri etrafında inşa edilmiştir. Playwright ile toplanan ham veriler, anlık olarak normalize edilip Apache Kafka aracılığıyla asenkron bir akışa dahil edilir. Bu mimari, sistemin hız limitlerine takılmasını engeller, olası kesintilerde veri kaybını sıfıra indirir ve uçtan uca güvenilir bir veri işleme süreci sunar.

> **Mimari Notu:** Sahibinden.com'un anti-bot koruması (Cloudflare/Kasada/DataDome) canlı kazıma için pratikte aşılamayacak kadar güçlü olduğundan bu platform kapsam dışı bırakılmıştır. Bunun yerine eğitim (train) verisi açık/herkese açık bir referans veri setinden, test/doğrulama verisi ise arabam.com'dan canlı kazınarak elde edilir.

Projenin temel misyonu; Türkiye otomobil piyasasındaki Otomobil, SUV, Minivan & Panelvan ve Elektrikli Araçlar kategorilerini analiz edip çapraz kaynak doğrulamalı (Cross-Source Validation) fiyat tahminleme modelleri (Machine Learning) için eğitim (Train) ve test (Validation/Test) kümeleri oluşturmaktır.

---

## ✨ Öne Çıkan Özellikler (Features)

*   **🛡️ Bot Engeli Toleransı:** Playwright ve Bright Data Browser API entegrasyonu sayesinde hedef platform üzerindeki temel anti-bot kontrolleri otonom olarak tolere edilir. (Not: Cloudflare/Kasada/DataDome gibi katmanlı korumaya sahip bazı platformlar canlı kazıma için kapsam dışıdır.)
*   **⚡ Ağ Trafiği ve Performans Optimizasyonu:** Tarayıcı seviyesinde alınan aksiyonlarla; resimler, reklamlar, fontlar ve gereksiz medya bileşenleri render edilmeden engellenir, böylece hız maksimize edilirken bant genişliği maliyeti minimize edilir.
*   **🎢 Apache Kafka ile Asenkron Veri İletimi:** Mesaj kuyruğu mekanizması bir "amortisör" görevi üstlenerek scraper'ın yüksek hızlı çıktılarını dengeler, bot engellemeleri anında dahi veri kaybını önler ve güvenli aktarım sağlar.
*   **🧹 Akıllı Metin Temizleme ve Standardizasyon:** Gelen düzensiz (unstructured) veriler (Örn: "750.000 TL" veya "68.000 km"), Regex tabanlı çalışan parser katmanıyla temizlenir. Canlı kazınan veri ile açık veri seti arasındaki sözlükler normalize edilerek kaynaklar arası veri uyumsuzluğu (Data Mismatch) kesin olarak çözülür.
*   **🎯 Çapraz Kaynak Doğrulaması (Cross-Source Validation):** Model over-fitting (aşırı öğrenme) riskini ortadan kaldırmak için eğitim verileri açık/herkese açık bir referans veri setinden (Train), test ve doğrulama verileri ise hedef ilan platformundan canlı kazınarak (Test/Val) elde edilir. Bu yaklaşım aynı zamanda modelin zaman içindeki piyasa kaymasına (distribution shift) karşı genelleme başarısını da sınar.

---

## 📁 Proje Dosya Yapısı

Sistemin esnek, modüler ve mikroservis mimarisine uygun klasör hiyerarşisi aşağıdaki gibidir:

```text
├── config/
│   ├── kafka-config.js
│   ├── proxy-config.js
│   └── scraper-rules.json
├── src/
│   ├── producers/
│   │   └── arabam-scraper.js
│   ├── consumers/
│   │   ├── kafka-consumer.js
│   │   └── csv-writer.js
│   ├── utils/
│   │   ├── json-parser.js
│   │   └── text-cleaner.js
│   └── app.js
├── data/
│   ├── raw/
│   └── output/
│       ├── train_dataset.csv
│       └── arabam_test_val.csv
├── ai-model/
│   ├── notebooks/
│   │   └── eda_and_preprocessing.ipynb
│   ├── prepare_train_dataset.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── README.md
├── package.json
└── requirements.txt
```

---

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
*   **Node.js**: v18 veya üzeri
*   **Python**: 3.10 veya üzeri
*   **Docker** ve **Docker Compose** (Apache Kafka ve Zookeeper altyapısı için)

### Adımlar
1.  Repoyu klonlayın: `git clone https://github.com/Mustafa-Aydin69/OtoMetrik.git`
2.  Node bağımlılıklarını kurun: `npm install`
3.  Python bağımlılıklarını kurun: `pip install -r requirements.txt`
4.  `.env.example` dosyasını `.env` olarak kopyalayıp Kafka/proxy değerlerini doldurun.
5.  Kafka altyapısını ayağa kaldırın: `docker compose up -d`
6.  Veri hattını başlatın: `npm start`

## 📊 Veri Kümesi Özellikleri (Dataset Schema)

Boru hattı üzerinden temizlenerek `data/output/` dizinine aktarılan CSV dosyaları aşağıdaki şema standartlarına sahiptir:

| Sütun Adı | Açıklama | Örnek Veri |
| :--- | :--- | :--- |
| `ilan_id` | Platformdaki benzersiz ilan numarası | `1105432901` |
| `arac_turu` | Aracın kategorisi | `Otomobil` |
| `marka` | Aracın markası | `Volkswagen` |
| `model` | Aracın spesifik modeli | `Golf` |
| `paket` | Modelin donanım/paket seviyesi | `R-Line` |
| `kasa_turu` | Aracın gövde tipi | `Hatchback` |
| `renk` | Aracın rengi | `Beyaz` |
| `motor_hacmi` | Motor hacmi ve tipi | `1.5 TSI` |
| `motor_gucu` | Motor gücü (hp) | `150` |
| `yil` | Üretim yılı | `2021` |
| `kilometre` | Aracın toplam kilometresi | `45000` |
| `yakit_turu` | Yakıt tipi | `Benzin` |
| `vites` | Şanzıman türü | `Yarı Otomatik` |
| `degisen_sayisi` | Değişen parça sayısı | `1` |
| `boyali_sayisi` | Boyalı parça sayısı | `2` |
| `fiyat` | Normalize edilmiş satış fiyatı (TL) | `1650000` |
| `scraped_at` | Verinin veritabanına/dosyaya yazıldığı ISO-8601 tarih damgası | `2026-06-30T16:35:00Z` |

---

## 🧠 Model Eğitim Süreci

Python katmanı, `data/output` altındaki temizlenmiş CSV dosyalarını kullanarak tahminleyici makine öğrenmesi modellerini inşa eder.

1.  **Veri Seti Hazırlığı:**
    `ai-model/prepare_train_dataset.py` betiği, açık/herkese açık referans veri setini indirip proje şemasına normalize ederek `data/output/train_dataset.csv` dosyasını üretir.
2.  **Keşifsel Veri Analizi (EDA):** 
    `ai-model/notebooks/eda_and_preprocessing.ipynb` dosyasında verilerin istatistiksel dağılımları ve korelasyonları incelenir, outlier (aykırı değer) analizleri yapılır.
3.  **Model Eğitimi:** 
    Eğitim süreci, modelin aşırı öğrenmesini (over-fitting) engellemek adına yalnızca `train_dataset.csv` verisiyle beslenir.
    ```bash
    python ai-model/train.py
    ```
4.  **Model Doğrulama ve Değerlendirme:** 
    Modelin daha önce görmediği, canlı kazınan güncel piyasa verisindeki genelleme başarısını sınamak için `arabam_test_val.csv` veri seti test/doğrulama için kullanılır. Performans aşağıdaki ana metriklerle ölçülür:
    ```bash
    python ai-model/evaluate.py
    ```
    *   **MAE (Mean Absolute Error):** Fiyat tahminlerindeki ortalama mutlak sapmayı ölçer.
    *   **RMSE (Root Mean Square Error):** Büyük hata paylarına duyarlı olarak modelin fiyat tahminlerindeki varyansını gösterir.
    *   **R² (R-Squared):** Bağımsız değişkenlerin (aracın özellikleri), bağımlı değişkeni (fiyat) ne oranda doğru açıklayabildiğini temsil eder.
