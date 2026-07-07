"""Text Cleaner (src/utils/text-cleaner.js'in Python karsiligi)."""
import re

TURKISH_MONTHS = {
    'ocak': 1, 'şubat': 2, 'subat': 2, 'mart': 3, 'nisan': 4, 'mayıs': 5, 'mayis': 5,
    'haziran': 6, 'temmuz': 7, 'ağustos': 8, 'agustos': 8, 'eylül': 9, 'eylul': 9,
    'ekim': 10, 'kasım': 11, 'kasim': 11, 'aralık': 12, 'aralik': 12,
}


def normalize_whitespace(value):
    return re.sub(r'\s+', ' ', str(value).strip())


def parse_number(value):
    if value is None:
        return None
    digits = re.sub(r'[^\d]', '', str(value))
    return int(digits) if digits else None


def parse_price_tl(value):
    return parse_number(value)


def parse_kilometre(value):
    return parse_number(value)


# "1 değişen, 2 boyalı" / "1 boyalı, 1 lokal boyalı" gibi birlesik metni degisen/boyali sayilarina ayirir.
# "Belirtilmemiş" bilinmiyor anlamina gelir (None); deger var ama bir tur hic gecmiyorsa o tur icin 0 demektir.
def parse_boya_degisen(value):
    if not value or value == 'Belirtilmemiş':
        return None, None

    degisen_match = re.search(r'(\d+)\s*değişen', value, re.IGNORECASE)
    boyali_matches = re.findall(r'(\d+)\s*(?:lokal\s*)?boyalı', value, re.IGNORECASE)

    degisen_sayisi = int(degisen_match.group(1)) if degisen_match else 0
    boyali_sayisi = sum(int(m) for m in boyali_matches)
    return degisen_sayisi, boyali_sayisi


# arabam.com motor hacmi/gucu bazen tek deger ("1461 cc", "90 hp"), bazen bucket araligi
# ("1601 - 1800 cm3", "76 - 100 HP") bazen de alt sinir ("1200 cm3' e kadar", "50 HP'ye kadar")
# olarak geliyor. Ucunu de ortak sayisal temsile (orta deger) cevirir; zaten sayisalsa oldugu gibi doner.
def parse_engine_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = normalize_whitespace(value)
    text = re.sub(r'cm3|cc|hp', '', text, flags=re.IGNORECASE)
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    if not numbers:
        return None

    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2
    if 'kadar' in text.lower():
        return numbers[0] / 2
    return float(numbers[0])


# arabam.com "Agir Hasarli" alani "Evet"/"Hayir" doner; digerleriyle (degisen_sayisi vb.)
# tutarli olmasi icin 1/0'a cevirir (JS tarafindaki parseEvetHayir ile ayni mantik).
def parse_evet_hayir(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == 'evet':
        return 1
    if normalized in ('hayır', 'hayir'):
        return 0
    return None


# "30 Haziran 2025" gibi Turkce tarihi ISO 8601'e cevirir; ayristirilamazsa None doner.
def parse_turkish_date(value):
    if not value:
        return None
    match = re.match(r'(\d{1,2})\s+(\S+)\s+(\d{4})', normalize_whitespace(value))
    if not match:
        return None
    day, month_name, year = match.groups()
    month = TURKISH_MONTHS.get(month_name.lower())
    if not month:
        return None
    return f'{year}-{month:02d}-{int(day):02d}T00:00:00.000Z'
