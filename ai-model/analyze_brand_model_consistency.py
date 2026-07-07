"""Faz 7 Madde 3: train/test/validation kaynaklari arasinda marka/model yazim tutarliligi raporu.

Uc kontrol yapar:
1. Ayni kaynak icinde case/bosluk farkiyla ayni degere karsilik gelen ham degerler
   (orn. "Mini" vs "MINI").
2. Kaynaklar arasi marka kumesi farki (hangi markalar sadece bir kaynakta var).
3. Ortak markalarda model/seri kumesi farki icin fuzzy-match taramasi: gercek yazim
   kaymasi (orn. "3 Serisi" vs "3.Serisi") ile gercek kapsam farkini (o modelin o
   kaynakta hic olmamasi) ayirt eder.
"""
import difflib
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
REPORT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'marka_model_uyum_raporu.txt')

SOURCES = {
    'train_dataset': {
        'path': os.path.join(BASE_DIR, 'data', 'output', 'train_dataset.csv'),
        'marka_col': 'marka', 'model_col': 'model',
    },
    'arabam_test_val': {
        'path': os.path.join(BASE_DIR, 'data', 'output', 'arabam_test_val.csv'),
        'marka_col': 'marka', 'model_col': 'model',
    },
    'cars1': {
        'path': os.path.join(BASE_DIR, 'kaggle', 'cars1.csv'),
        'marka_col': 'marka', 'model_col': 'seri',
    },
}


def load_sources():
    return {name: pd.read_csv(spec['path'], low_memory=False, encoding='utf-8-sig')
            for name, spec in SOURCES.items()}


def case_whitespace_dupes(series):
    normalized = {}
    for v in series.dropna().unique():
        key = str(v).strip().lower()
        normalized.setdefault(key, set()).add(v)
    return {k: v for k, v in normalized.items() if len(v) > 1}


def check_case_whitespace(dfs, lines):
    lines.append('=== 1) Ayni kaynak icinde case/bosluk cakismalari ===')
    any_found = False
    for name, spec in SOURCES.items():
        df = dfs[name]
        for field, col in [('marka', spec['marka_col']), ('model', spec['model_col'])]:
            dupes = case_whitespace_dupes(df[col])
            for key, variants in dupes.items():
                any_found = True
                lines.append(f'{name}.{field}: {sorted(variants)}')
    if not any_found:
        lines.append('Bulunamadi.')
    lines.append('')


def check_brand_sets(dfs, lines):
    lines.append('=== 2) Kaynaklar arasi marka kume farki ===')
    brand_sets = {
        name: set(dfs[name][spec['marka_col']].dropna().str.strip())
        for name, spec in SOURCES.items()
    }
    names = list(brand_sets)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            a, b = names[i], names[j]
            only_a = sorted(brand_sets[a] - brand_sets[b])
            lines.append(f'{a} - {b} ({len(only_a)} marka): {only_a}')
    lines.append('')


def check_model_spelling_drift(dfs, lines):
    lines.append('=== 3) Ortak markalarda model/seri fuzzy-match taramasi (yazim kaymasi mi, kapsam farki mi) ===')
    df1, df3 = dfs['train_dataset'], dfs['cars1']
    common_brands = set(df1['marka'].dropna()) & set(df3['marka'].dropna())

    found_any = False
    for marka in sorted(common_brands):
        s1 = set(df1[df1['marka'] == marka]['model'].dropna().str.strip())
        s3 = set(df3[df3['marka'] == marka]['seri'].dropna().str.strip())
        only1, only3 = s1 - s3, s3 - s1
        for a in only1:
            matches = difflib.get_close_matches(a, only3, n=1, cutoff=0.75)
            if matches:
                found_any = True
                lines.append(f'{marka}: {a!r} <-> {matches[0]!r} (olasi yazim kaymasi)')
    if not found_any:
        lines.append('Yakin-yazim eslesme bulunamadi -> tum farklar gercek kapsam farki (model o kaynakta yok), yazim tutarsizligi degil.')
    lines.append('')


def main():
    dfs = load_sources()
    lines = []
    check_case_whitespace(dfs, lines)
    check_brand_sets(dfs, lines)
    check_model_spelling_drift(dfs, lines)

    report = '\n'.join(lines)
    print(report)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nRapor dosyaya yazildi: {REPORT_PATH}')


if __name__ == '__main__':
    main()
