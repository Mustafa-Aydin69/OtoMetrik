/**
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

export const YEAR_MIN = 1950;
export const YEAR_MAX = 2027;

export const MILEAGE_MIN = 0;
export const MILEAGE_MAX = 1000000;

export const ENGINE_POWER_MIN = 1;
export const ENGINE_POWER_MAX = 2000;

export const ENGINE_DISPLACEMENT_MIN = 0;
export const ENGINE_DISPLACEMENT_MAX = 9000;

export const REPLACED_PARTS_MIN = 0;
export const REPLACED_PARTS_MAX = 13;

export const PAINTED_PARTS_MIN = 0;
export const PAINTED_PARTS_MAX = 13;

export const MAX_TOTAL_DAMAGED_PARTS = 13;
