/**
 * Basit process-içi TTL cache. generation-resolver.ts ve image-provider.ts
 * tarafından paylaşılır — aynı marka/model için Wikipedia/Commons'a tekrar
 * tekrar istek atılmasını önler (bkz. istenen "performans" gereksinimi).
 *
 * Not: Bu bellek-içi bir cache'tir, tek bir sunucu sürecinin ömrüyle
 * sınırlıdır (yeniden deploy/restart'ta sıfırlanır, çoklu instance'lar
 * arasında paylaşılmaz). Gerçek çok-instance production için Redis/KV
 * gibi paylaşılan bir depoya taşınabilir — arayüz (get/set) aynı kalır.
 */

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

const store = new Map<string, CacheEntry<unknown>>();

export async function cached<T>(
  key: string,
  ttlMs: number,
  compute: () => Promise<T>
): Promise<T> {
  const hit = store.get(key);
  if (hit && hit.expiresAt > Date.now()) {
    return hit.value as T;
  }
  const value = await compute();
  store.set(key, { value, expiresAt: Date.now() + ttlMs });
  return value;
}

export const CACHE_TTL = {
  GENERATION: 7 * 24 * 60 * 60 * 1000, // 7 gün — nesil zaman çizelgesi neredeyse hiç değişmez
  IMAGE_SEARCH: 24 * 60 * 60 * 1000, // 1 gün — Commons'a yeni foto eklenebilir, çok uzun tutulmaz
} as const;
