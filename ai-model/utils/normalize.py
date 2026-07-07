"""Faz 7 Madde 4: uc kaynagi (train_dataset, arabam_test_val, cars1) ortak temsile ceviren
normalizasyon modulu. Madde 1-3'te bulunan somut uyusmazliklar icin yazildi; genel/varsayimsal
bir "fuzzy normalizer" degil - bulunanlarin disinda bir seyi degistirmez:

- kasa_turu: "-" placeholder'i eksik deger (None) olarak ele alinir.
- marka/model: ayni kaynak icinde bulunan case cakismalari kanonik forma indirilir
  (Mini/MINI -> Mini, I30/i30 -> i30; kanonik form cars1 ve coğunluktaki yazimla uyumlu).
- motor_hacmi/motor_gucu: tekil/aralik/ust-sinir metinleri ortak sayisal temsile cevrilir
  (bkz. text_cleaner.parse_engine_value).
"""
import pandas as pd

from .text_cleaner import normalize_whitespace, parse_engine_value

MARKA_ALIASES = {
    'mini': 'Mini',
}

MODEL_ALIASES = {
    'i30': 'i30',
}


def _is_missing(value):
    return value is None or (isinstance(value, float) and pd.isna(value))


def normalize_marka(value):
    if _is_missing(value):
        return None
    text = normalize_whitespace(value)
    return MARKA_ALIASES.get(text.lower(), text)


def normalize_model(value):
    if _is_missing(value):
        return None
    text = normalize_whitespace(value)
    return MODEL_ALIASES.get(text.lower(), text)


def normalize_kasa_turu(value):
    if _is_missing(value):
        return None
    text = normalize_whitespace(value)
    return None if text == '-' else text


def normalize_dataframe(df, marka_col='marka', model_col='model',
                         kasa_col='kasa_turu', hacmi_col='motor_hacmi', gucu_col='motor_gucu'):
    df = df.copy()
    df[marka_col] = df[marka_col].apply(normalize_marka)
    df[model_col] = df[model_col].apply(normalize_model)
    df[kasa_col] = df[kasa_col].apply(normalize_kasa_turu)
    df[hacmi_col] = df[hacmi_col].apply(parse_engine_value)
    df[gucu_col] = df[gucu_col].apply(parse_engine_value)
    return df
