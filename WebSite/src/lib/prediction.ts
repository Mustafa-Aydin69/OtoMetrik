/**
 * Client tarafı tahmin servisi: formu /api/predict'e gönderir.
 * Gerçek model entegrasyonu server tarafında (route.ts) PREDICTION_API_URL
 * environment variable'ı üzerinden yapılır — client bu ayrımı bilmez.
 */
import type { PredictionInput } from "./validation";

export interface PredictionResponse {
  price: number;
  currency: "TRY";
  source: "model" | "mock";
}

export async function requestPrediction(
  input: PredictionInput
): Promise<PredictionResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) {
    let message = "Tahmin isteği başarısız oldu. Lütfen tekrar deneyin.";
    try {
      const data = (await res.json()) as { error?: string };
      if (data.error) message = data.error;
    } catch {
      // gövde JSON değilse varsayılan mesaj kalır
    }
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
}

export function formatTRY(price: number): string {
  return new Intl.NumberFormat("tr-TR", {
    style: "currency",
    currency: "TRY",
    maximumFractionDigits: 0,
  }).format(price);
}
