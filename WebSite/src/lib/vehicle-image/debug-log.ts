/**
 * Basit yapılandırılmış logger - vehicle-image pipeline'ının debug çıktısı
 * için (bkz. plan Madde 19-20 - "structured logs, production UI'ya çıkmak
 * zorunda değil, dev console/log yeterli"). Yalnızca development'ta veya
 * OTOMETRIK_DEBUG=1 iken konsola yazar; production'da sessiz. Python
 * tarafındaki serve.py'nin OTOMETRIK_DEBUG deseniyle tutarlı.
 */
const DEBUG_ENABLED = process.env.NODE_ENV !== "production" || process.env.OTOMETRIK_DEBUG === "1";

export function debugLog(event: string, data: Record<string, unknown>): void {
  if (!DEBUG_ENABLED) return;
  console.debug(`[vehicle-image:${event}]`, JSON.stringify(data, null, 2));
}
