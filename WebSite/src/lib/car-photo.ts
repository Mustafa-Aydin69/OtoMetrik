/**
 * Client tarafı: /api/car-image'a tahmin girdisini gönderip aday araç
 * fotoğrafı URL'lerini alır. Fiyat tahminine paralel, en spesifikten en
 * genele sıralı bir liste döner; ilk çalışan URL PredictionResult'ta
 * gösterilir (bkz. o bileşendeki onError kademe düşürme mantığı).
 * Hata durumunda sessizce boş dizi döner — bu özellik opsiyoneldir,
 * fiyat tahminini asla bloklamamalı.
 */
import type { PredictionInput } from "./validation";

export async function requestCarImageUrls(input: PredictionInput): Promise<string[]> {
  try {
    const res = await fetch("/api/car-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        brand: input.brand,
        model: input.model,
        year: input.year,
        trim: input.trim,
        color: input.color,
      }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { imageUrls?: string[] };
    return Array.isArray(data.imageUrls) ? data.imageUrls : [];
  } catch {
    return [];
  }
}
