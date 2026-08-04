"""Faz 13: website (kullanici dostu Turkce etiket) ile model artefaktinin
egitim-zamani kanonik kategori degerleri arasindaki TEK dogruluk kaynagi.

Motivasyon (bkz. dagitima hazirlik analizi): website'in elle yazilmis dropdown
listeleri (WebSite/src/lib/validation.ts) ile modelin egitim-zamaninda gordugu
gercek kategori string'leri (bu dosyadaki LABEL_TO_CANONICAL degerleri,
models/lightgbm_final.joblib icindeki category_sets'ten dogrulanir) arasinda
birebir string uyuşmazliklari vardi (orn. "Manuel" vs "Düz", "Mercedes-Benz"
vs "Mercedes - Benz"). apply_saved_categories() bilinmeyen bir kategoriyi
sessizce NaN'a ceviriyordu - bu, olculebilir (%11-27) fiyat sapmalarina yol
aciyordu.

Bu modul TEK yerde tutulur (Python tarafinda hem serve.py hem
generate_category_mapping.py buradan okur; generate_category_mapping.py
WebSite/src/lib/category-options.generated.ts dosyasini BURADAN turetir -
website tarafinda ayrica elle yazilmis bir liste YOKTUR).

Her alan icin LABEL_TO_CANONICAL: kullaniciya gosterilen etiket -> modelin
gordugu kanonik deger.
    Bos string ("") kanonik deger = KASITLI "bilinmiyor" sentinel'i (bkz. renk
    alanindaki "Diğer" -> "Diğer" farkli, o gercek bir kategori, sentinel
    degil). marka alaninda artik boyle bir sentinel YOK - marka listesi
    model artefaktinin gordugu TUM 68 kanonik degeri kapsiyor (bkz. asagidaki
    marka sozlugu), kullanicinin "bilinmiyor" secebilecegi bir durum kalmadi.
    "" hicbir egitim kategorisiyle eslesmedigi icin apply_saved_categories()
    tarafindan dogal olarak NaN'a dusurulur - bu DOGRU davranistir (model
    gercekten bilinmeyen bir markayi tahmin edemez, bu bir hata degildir).
    resolve_canonical() bu degeri 422 ILE REDDETMEZ (kasitli secim, gecerli).
    Gercekten TANINMAYAN (ne etiket ne kanonik olarak eslenen) bir deger ise
    422 ile reddedilir (bkz. serve.py resolve_canonical() kullanimi).

UNSUPPORTED_LABELS: kullanici tarafinda dusunulebilecek ama modelin egitim
verisinde guvenli/tekil bir kanonik karsiligi OLMAYAN etiketler. Bunlar
otomatik tahmin edilip eslenmez (belirsiz eslesme riski) - formdan cikarilir
ve nedeni burada acikca kayit altina alinir.
"""

CANONICAL_FIELDS = ["marka", "vites", "yakit_turu", "kasa_turu", "renk"]

LABEL_TO_CANONICAL = {
    "vites": {
        "Manuel": "Düz",
        "Otomatik": "Otomatik",
        "Yarı Otomatik": "Yarı Otomatik",
    },
    "yakit_turu": {
        "Benzin": "Benzin",
        "Dizel": "Dizel",
        "LPG": "LPG & Benzin",
        "Hibrit": "Hibrit",
        "Elektrik": "Elektrik",
    },
    # Model artefaktinin gordugu TUM 68 kanonik marka degeri (bkz. Faz 24 -
    # onceden yalnizca 19 marka + "Diğer" sentinel'i vardi, eğitim verisinde
    # gorulen 49 marka - Mazda, Mini, Porsche, Jeep, Alfa Romeo, Tesla, TOGG
    # dahil - UI'dan hic secilemiyordu). Kaynak: load_model()['category_sets']
    # ['marka'] (models/lightgbm_final.joblib), etikete gore alfabetik siralanmis.
    # Kanonikten farkli etiket sadece 3 markada: Citroën, Mercedes-Benz,
    # Iveco-Otoyol (gorsel/yazim duzeltmesi, egitim verisindeki ham string
    # farkli).
    "marka": {
        "Alfa Romeo": "Alfa Romeo",
        "Aston Martin": "Aston Martin",
        "Audi": "Audi",
        "Bentley": "Bentley",
        "BMW": "BMW",
        "Buick": "Buick",
        "BYD": "BYD",
        "Cadillac": "Cadillac",
        "Chery": "Chery",
        "Chevrolet": "Chevrolet",
        "Chrysler": "Chrysler",
        "Citroën": "Citroen",
        "Cupra": "Cupra",
        "Dacia": "Dacia",
        "Daewoo": "Daewoo",
        "Daihatsu": "Daihatsu",
        "Dodge": "Dodge",
        "DS Automobiles": "DS Automobiles",
        "Fiat": "Fiat",
        "Ford": "Ford",
        "Geely": "Geely",
        "Honda": "Honda",
        "Hyundai": "Hyundai",
        "Ikco": "Ikco",
        "Infiniti": "Infiniti",
        "Isuzu": "Isuzu",
        "Iveco-Otoyol": "Iveco - Otoyol",
        "Jaecoo": "Jaecoo",
        "Jaguar": "Jaguar",
        "Jeep": "Jeep",
        "Kia": "Kia",
        "Lada": "Lada",
        "Lancia": "Lancia",
        "Land Rover": "Land Rover",
        "Lexus": "Lexus",
        "Lincoln": "Lincoln",
        "Lotus": "Lotus",
        "Maserati": "Maserati",
        "Maxus": "Maxus",
        "Mazda": "Mazda",
        "Mercedes-Benz": "Mercedes - Benz",
        "Mercury": "Mercury",
        "MG": "MG",
        "Mini": "Mini",
        "Mitsubishi": "Mitsubishi",
        "Nissan": "Nissan",
        "Opel": "Opel",
        "Peugeot": "Peugeot",
        "Pontiac": "Pontiac",
        "Porsche": "Porsche",
        "Proton": "Proton",
        "Renault": "Renault",
        "Rolls-Royce": "Rolls-Royce",
        "Rover": "Rover",
        "Saab": "Saab",
        "Seat": "Seat",
        "Skoda": "Skoda",
        "Smart": "Smart",
        "Ssangyong": "Ssangyong",
        "Subaru": "Subaru",
        "Suzuki": "Suzuki",
        "Tata": "Tata",
        "Tesla": "Tesla",
        "Tofaş": "Tofaş",
        "TOGG": "TOGG",
        "Toyota": "Toyota",
        "Volkswagen": "Volkswagen",
        "Volvo": "Volvo",
    },
    "renk": {
        "Siyah": "Siyah",
        "Beyaz": "Beyaz",
        "Gri": "Gri",
        "Gümüş": "Gri (Gümüş)",
        "Mavi": "Mavi",
        "Kırmızı": "Kırmızı",
        "Kahverengi": "Kahverengi",
        "Bej": "Bej",
        "Yeşil": "Yeşil",
        "Turuncu": "Turuncu",
        "Sarı": "Sarı",
        "Bordo": "Bordo",
        "Lacivert": "Lacivert",
        "Diğer": "Diğer",
    },
    "kasa_turu": {
        "Sedan": "Sedan",
        "Hatchback (3 Kapı)": "Hatchback/3",
        "Hatchback (5 Kapı)": "Hatchback/5",
        "SUV": "SUV",
        "Coupe": "Coupe",
        "Station Wagon": "Station wagon",
        "Cabrio": "Cabrio",
        "Pick-up": "Pick-up",
        "MPV": "MPV",
        "Panelvan": "Panel Van",
        "Crossover": "Crossover",
    },
}

UNSUPPORTED_LABELS = {
    "kasa_turu": {
        "Hatchback": (
            "Eğitim verisinde 'Hatchback/3' ve 'Hatchback/5' ayrı kategoriler; "
            "tekil 'Hatchback' belirsiz olduğu için otomatik eşlenmedi (kapı "
            "sayısı sorulmalı)."
        ),
        "Minivan": (
            "Modelin eğitim verisinde 'Minivan' kategorisi yok; en yakın aday "
            "'Minibüs' semantik olarak farklı bir araç sınıfı (yolcu minibüsü) "
            "olduğu için otomatik eşlenmedi. Yeniden eğitim/veri toplama "
            "gerektiren desteklenmeyen seçenek."
        ),
    },
}


def resolve_canonical(field, raw_value, category_sets):
    """raw_value -> (basarili_mi, kanonik_deger).

    Kabul sirasi: (1) bilinen kullanici etiketi (LABEL_TO_CANONICAL) - kanonigi
    "" ise kasitli-bilinmiyor olarak basarili sayilir (bkz. modul docstring'i);
    (2) zaten kanonik bir deger (dogrudan API entegrasyonlari/testler icin);
    (3) hicbiri degilse basarisiz -> cagiran 422 dondurmeli.
    """
    field_labels = LABEL_TO_CANONICAL.get(field, {})
    if raw_value in field_labels:
        return True, field_labels[raw_value]

    valid_canonical = category_sets.get(field)
    if valid_canonical is not None and raw_value in valid_canonical:
        return True, raw_value

    return False, None


def allowed_examples(field, category_sets, limit=5):
    values = list(category_sets.get(field, []))
    return values[:limit]


def check_drift(category_sets):
    """LABEL_TO_CANONICAL'daki her GERCEK (bos-string kasitli-bilinmiyor
    sentinel'i haric) kanonik degerin, modelin GERCEKTEN egitildigi kategori
    kumesinde var oldugunu dogrular. Model yeniden egitilip kategori kumesi
    degisirse (orn. bir kategori artik kullanilmiyorsa) bu mapping'in sessizce
    bozulmasini onler."""
    issues = []
    for field, labels in LABEL_TO_CANONICAL.items():
        valid = set(category_sets.get(field, []))
        for label, canonical in labels.items():
            if canonical != "" and canonical not in valid:
                issues.append(
                    f'{field}: "{label}" -> "{canonical}" ama "{canonical}" '
                    f"artik modelin kategori kumesinde yok"
                )
    return issues


def build_frontend_export(category_sets):
    """Website'in category-options.generated.ts dosyasi icin JSON-serilestirilebilir
    yapi: her alan icin {label, value} listesi (kasitli-bilinmiyor haric, cunku
    frontend'in her zaman gonderebilecegi bir deger olmali) + desteklenmeyen
    etiketlerin nedenleri."""
    export = {}
    for field, labels in LABEL_TO_CANONICAL.items():
        options = [{"label": label, "value": canonical}
                   for label, canonical in labels.items()]
        export[field] = {
            "options": options,
            "unsupported": UNSUPPORTED_LABELS.get(field, {}),
        }
    return export
