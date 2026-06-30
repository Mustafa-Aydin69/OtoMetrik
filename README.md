# 🚀 AutoFlow AI: Uçtan Uca Araç Fiyat Tahminleme ve Veri Hattı

**AutoFlow AI**, Sahibinden.com ve Arabam.com platformlarından yüksek performanslı asenkron web kazıma teknikleriyle piyasa verilerini toplayan, temizleyen ve makine öğrenmesi modelleri için yapılandırılmış veri setlerine dönüştüren gelişmiş bir büyük veri hattı (Data Pipeline) projesidir.

Sistem; birbirinden tamamen izole edilmiş (decoupled) mikroservis bileşenleri etrafında inşa edilmiştir. Playwright ile toplanan ham veriler, anlık olarak normalize edilip Apache Kafka aracılığıyla asenkron bir akışa dahil edilir. Bu mimari, sistemin hız limitlerine takılmasını engeller, olası kesintilerde veri kaybını sıfıra indirir ve uçtan uca güvenilir bir veri işleme süreci sunar.

Projenin temel misyonu; Türkiye otomobil piyasasındaki Otomobil, SUV, Minivan & Panelvan ve Elektrikli Araçlar kategorilerini analiz edip çapraz platform doğrulamalı (Cross-Domain Validation) fiyat tahminleme modelleri (Machine Learning) için eğitim (Train) ve test (Validation/Test) kümeleri oluşturmaktır.

---

## ✨ Öne Çıkan Özellikler (Features)

*   **🛡️ Otonom Bot Engeli Aşma:** Playwright ve Bright Data Browser API entegrasyonu sayesinde Cloudflare, Kasada ve DataDome gibi karmaşık anti-bot sistemleri otonom olarak aşılır.
*   **⚡ Ağ Trafiği ve Performans Optimizasyonu:** Tarayıcı seviyesinde alınan aksiyonlarla; resimler, reklamlar, fontlar ve gereksiz medya bileşenleri render edilmeden engellenir, böylece hız maksimize edilirken bant genişliği maliyeti minimize edilir.
*   **🎢 Apache Kafka ile Asenkron Veri İletimi:** Mesaj kuyruğu mekanizması bir "amortisör" görevi üstlenerek scraper'ın yüksek hızlı çıktılarını dengeler, bot engellemeleri anında dahi veri kaybını önler ve güvenli aktarım sağlar.
*   **🧹 Akıllı Metin Temizleme ve Standardizasyon:** Gelen düzensiz (unstructured) veriler (Örn: "750.000 TL" veya "68.000 km"), Regex tabanlı çalışan parser katmanıyla temizlenir. Her iki platformdaki veri sözlükleri normalize edilerek platformlar arası veri uyumsuzluğu (Data Mismatch) kesin olarak çözülür.
*   **🎯 Çapraz Platform Doğrulaması (Cross-Domain Validation):** Model over-fitting (aşırı öğrenme) riskini ortadan kaldırmak için eğitim verileri Sahibinden.com platformundan (Train), test ve doğrulama verileri ise Arabam.com platformundan (Test/Val) çekilerek sağlam bir makine öğrenmesi hattı kurulur.

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
│   │   ├── sahibinden-scraper.js
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
│       ├── sahibinden_train.csv
│       └── arabam_test_val.csv
├── ai-model/
│   ├── notebooks/
│   │   └── eda_and_preprocessing.ipynb
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

### Adım Adım Kurulum

**1. Depoyu Klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADI/autoflow-ai.git
cd autoflow-ai
```

**2. Node.js Bağımlılıklarını Yükleyin (Data Pipeline):**
```bash
npm install
```

**3. Python Bağımlılıklarını Yükleyin (AI Model):**
```bash
pip install -r requirements.txt
```

**4. Apache Kafka'yı Ayağa Kaldırın:**
Projedeki Kafka altyapısını başlatmak için kök dizinde veya genel Docker komutlarıyla bir Kafka cluster'ı başlatın:
```bash
docker run -d --name zookeeper -p 2181:2181 zookeeper:latest
docker run -d --name kafka -p 9092:9092 \
    -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
    confluentinc/cp-kafka:latest
```
*(Alternatif olarak, projenize bir `docker-compose.yml` ekleyip `docker-compose up -d` komutuyla da ayağa kaldırabilirsiniz.)*

**5. Projeyi Başlatın:**
Tüm veri kazıma, kuyruğa alma ve CSV yazma pipeline'ını asenkron olarak tetiklemek için ana uygulamayı çalıştırın:
```bash
node src/app.js
```

---

## 📊 Veri Kümesi Özellikleri (Dataset Schema)

Boru hattı üzerinden temizlenerek `data/output/` dizinine aktarılan CSV dosyaları aşağıdaki şema standartlarına sahiptir:

| Sütun Adı | Açıklama | Örnek Veri |
| :--- | :--- | :--- |
| `ilan_id` | Platformdaki benzersiz ilan numarası | `1105432901` |
| `arac_turu` | Aracın kategorisi | `Otomobil` |
| `marka` | Aracın markası | `Volkswagen` |
| `model` | Aracın spesifik modeli | `Golf 1.5 TSI R-Line` |
| `yil` | Üretim yılı | `2021` |
| `kilometre` | Aracın toplam kilometresi | `45000` |
| `yakit_turu` | Yakıt tipi | `Benzin` |
| `vites` | Şanzıman türü | `Yarı Otomatik` |
| `konum` | İlanın verildiği lokasyon (İl/İlçe) | `İstanbul / Kadıköy` |
| `fiyat` | Normalize edilmiş satış fiyatı (TL) | `1650000` |
| `scraped_at` | Verinin veritabanına/dosyaya yazıldığı ISO-8601 tarih damgası | `2026-06-30T16:35:00Z` |

---

## 🧠 Model Eğitim Süreci

Python katmanı, `data/output` altındaki temizlenmiş CSV dosyalarını kullanarak tahminleyici makine öğrenmesi modellerini inşa eder.

1.  **Keşifsel Veri Analizi (EDA):** 
    `ai-model/notebooks/eda_and_preprocessing.ipynb` dosyasında verilerin istatistiksel dağılımları ve korelasyonları incelenir, outlier (aykırı değer) analizleri yapılır.
2.  **Model Eğitimi:** 
    Eğitim süreci, modelin aşırı öğrenmesini (over-fitting) engellemek adına yalnızca `sahibinden_train.csv` verisiyle beslenir.
    ```bash
    python ai-model/train.py
    ```
3.  **Model Doğrulama ve Değerlendirme:** 
    Modelin daha önce görmediği platform verilerindeki genelleme başarısını sınamak için `arabam_test_val.csv` veri seti test/doğrulama için kullanılır. Performans aşağıdaki ana metriklerle ölçülür:
    ```bash
    python ai-model/evaluate.py
    ```
    *   **MAE (Mean Absolute Error):** Fiyat tahminlerindeki ortalama mutlak sapmayı ölçer.
    *   **RMSE (Root Mean Square Error):** Büyük hata paylarına duyarlı olarak modelin fiyat tahminlerindeki varyansını gösterir.
    *   **R² (R-Squared):** Bağımsız değişkenlerin (aracın özellikleri), bağımlı değişkeni (fiyat) ne oranda doğru açıklayabildiğini temsil eder.
