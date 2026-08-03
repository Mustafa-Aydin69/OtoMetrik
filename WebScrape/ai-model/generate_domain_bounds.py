"""Faz 14: domain_validation.py'deki TEK kaynaktan WebSite/src/lib/domain-bounds.generated.ts'yi
uretir - website'in sayisal alan sinirlari (yil/kilometre/motor_gucu/motor_hacmi/
degisen_sayisi/boyali_sayisi + toplam hasarli parca siniri) elle YAZILMAZ, buradan
turetilir. Backend (serve.py + domain_validation.py) HER ZAMAN nihai otoritedir -
bu dosya sadece frontend'in ayni sayilarla erken/UX-dostu geri bildirim vermesi icindir.

Calistirma (ai-model/ calisma dizini olarak):
    python generate_domain_bounds.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from domain_validation import FIELD_BOUNDS, MAX_TOTAL_DAMAGED_PARTS

WEBSITE_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'WebSite', 'src', 'lib', 'domain-bounds.generated.ts'
)

FIELD_TO_TS_CONST = {
    'yil': 'YEAR',
    'kilometre': 'MILEAGE',
    'motor_gucu': 'ENGINE_POWER',
    'motor_hacmi': 'ENGINE_DISPLACEMENT',
    'degisen_sayisi': 'REPLACED_PARTS',
    'boyali_sayisi': 'PAINTED_PARTS',
}

HEADER = """/**
 * OTOMATIK URETILMISTIR - ELLE DUZENLEMEYIN.
 *
 * Kaynak: WebScrape/ai-model/domain_validation.py (TEK dogruluk kaynagi -
 * egitim verisinin gercek yuzdelik dagilimlarindan turetilir, bkz. o
 * dosyadaki gerekce yorumlari). Uretmek icin:
 *   cd WebScrape/ai-model && python generate_domain_bounds.py
 *
 * Backend (serve.py) HER ZAMAN nihai otoritedir - bu sabitler yalnizca
 * frontend'in ayni sinirlarla ERKEN geri bildirim vermesi icindir.
 */
"""


def main():
    lines = [HEADER]
    for field, const_prefix in FIELD_TO_TS_CONST.items():
        bounds = FIELD_BOUNDS[field]
        lines.append(f'export const {const_prefix}_MIN = {bounds["min"]};')
        lines.append(f'export const {const_prefix}_MAX = {bounds["max"]};\n')
    lines.append(f'export const MAX_TOTAL_DAMAGED_PARTS = {MAX_TOTAL_DAMAGED_PARTS};\n')

    output_path = os.path.abspath(WEBSITE_OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(f'uretildi: {output_path}')


if __name__ == '__main__':
    main()
