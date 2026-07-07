"""Faz 7 Madde 1: train/test/validation kaynaklari arasinda kategorik alan uyum raporu.

yakit_turu, vites, kasa_turu alanlarinin her kaynaktaki benzersiz degerlerini
cikarir ve uc kaynak arasinda karsilastirir (hangi degerler hangi kaynak(lar)da var).
"""
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'kategorik_uyum_raporu.txt')

SOURCES = {
    'train_dataset': {
        'path': os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv'),
        'columns': {'yakit_turu': 'yakit_turu', 'vites': 'vites', 'kasa_turu': 'kasa_turu'},
    },
    'arabam_test_val': {
        'path': os.path.join(BASE_DIR, 'data', 'output', 'arabam_test_val.csv'),
        'columns': {'yakit_turu': 'yakit_turu', 'vites': 'vites', 'kasa_turu': 'kasa_turu'},
    },
    'cars1': {
        'path': os.path.join(BASE_DIR, 'kaggle', 'cars1.csv'),
        'columns': {'yakit_turu': 'yakit_tipi', 'vites': 'vites_tipi', 'kasa_turu': 'kasa_tipi'},
    },
}

FIELDS = ['yakit_turu', 'vites', 'kasa_turu']


def load_unique_values():
    result = {field: {} for field in FIELDS}
    for source_name, spec in SOURCES.items():
        df = pd.read_csv(spec['path'], encoding='utf-8-sig')
        for field in FIELDS:
            col = spec['columns'][field]
            counts = df[col].value_counts(dropna=False)
            result[field][source_name] = counts
    return result


def build_report(unique_values):
    lines = []
    for field in FIELDS:
        lines.append('=' * 70)
        lines.append(f'ALAN: {field}')
        lines.append('=' * 70)

        per_source = unique_values[field]
        all_values = set()
        for counts in per_source.values():
            all_values.update(counts.index)

        rows = []
        for value in all_values:
            presence = []
            for source_name, counts in per_source.items():
                if value in counts.index:
                    presence.append(source_name)
            rows.append((value, presence, len(presence)))

        # Once kac kaynakta gorulduguyle sirala (once ortaklar, sonra tekiller), sonra deger adiyla.
        rows.sort(key=lambda r: (-r[2], str(r[0])))

        lines.append(f'{"deger":40s} {"kac kaynakta":15s} kaynaklar')
        lines.append('-' * 70)
        for value, presence, n in rows:
            lines.append(f'{str(value):40s} {n:<15d} {", ".join(presence)}')

        mismatches = [r for r in rows if r[2] < len(per_source)]
        lines.append('')
        lines.append(f'-> {len(all_values)} benzersiz deger, {len(mismatches)} tanesi tum kaynaklarda ortak degil.')
        lines.append('')
    return '\n'.join(lines)


def main():
    unique_values = load_unique_values()
    report = build_report(unique_values)
    print(report)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nRapor dosyaya yazildi: {REPORT_PATH}')


if __name__ == '__main__':
    main()
