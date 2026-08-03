"""Faz 13: category_mapping.py'deki TEK kaynaktan WebSite/src/lib/category-options.generated.ts'yi
uretir. Website tarafinda kategori secenekleri (marka/vites/yakit_turu/kasa_turu/renk)
elle YAZILMAZ - bu script'in ciktisidir. category_mapping.py'de bir kategori
degisirse (yeni marka eklenir, bir etiket duzeltilir, vb.) sadece bu script
yeniden calistirilir; WebSite dosyasi elle duzenlenmez.

Calistirma (ai-model/ calisma dizini olarak):
    python generate_category_mapping.py
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from category_mapping import LABEL_TO_CANONICAL, UNSUPPORTED_LABELS, check_drift
from train import load_model

WEBSITE_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'WebSite', 'src', 'lib', 'category-options.generated.ts'
)

HEADER = """/**
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
"""


def ts_string_literal(value):
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def render_ts(export):
    lines = [HEADER]
    for field, options in export.items():
        const_name = f'{field.upper()}_OPTIONS'
        lines.append(f'export const {const_name}: CategoryOption[] = [')
        for label, canonical in options.items():
            lines.append(
                f'  {{ label: {ts_string_literal(label)}, value: {ts_string_literal(canonical)} }},'
            )
        lines.append('];\n')
    return '\n'.join(lines)


def main():
    artifact = load_model()
    category_sets = {k: list(v) for k, v in artifact['category_sets'].items()}

    issues = check_drift(category_sets)
    if issues:
        print('DRIFT BULUNDU - uretim durduruldu, once category_mapping.py duzeltilmeli:')
        for issue in issues:
            print(' -', issue)
        raise SystemExit(1)

    ts_content = render_ts(LABEL_TO_CANONICAL)
    output_path = os.path.abspath(WEBSITE_OUTPUT_PATH)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(ts_content)
    print(f'uretildi: {output_path}')

    for field, labels in LABEL_TO_CANONICAL.items():
        unsupported = UNSUPPORTED_LABELS.get(field, {})
        print(f'{field}: {len(labels)} secenek, {len(unsupported)} desteklenmeyen etiket')
        for label, reason in unsupported.items():
            print(f'  [desteklenmiyor] {label}: {reason}')


if __name__ == '__main__':
    main()
