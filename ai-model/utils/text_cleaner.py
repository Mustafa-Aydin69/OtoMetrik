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
