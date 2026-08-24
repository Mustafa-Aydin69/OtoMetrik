import { NextResponse } from "next/server";
import { coercePrediction, toCanonicalPayload } from "@/lib/validation";
import { mockPredict } from "@/lib/mock-prediction";

/**
 * POST /api/predict
 * PREDICTION_API_URL tanımlıysa isteği gerçek modele iletir,
 * değilse deterministik mock tahmin döner (development).
 */
export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Geçersiz istek: JSON gövdesi okunamadı." },
      { status: 400 }
    );
  }

  const { input, errors } = coercePrediction(body);
  if (!input) {
    return NextResponse.json(
      { error: "Form verileri geçersiz.", fields: errors },
      { status: 400 }
    );
  }

  const apiUrl = process.env.PREDICTION_API_URL;

  if (apiUrl) {
    try {
      // Model servisi kanonik kategori değerleri bekler (örn. "Düz", "Manuel"
      // değil) - bkz. lib/validation.ts toCanonicalPayload().
      const canonicalInput = toCanonicalPayload(input);
      // new URL("/predict", apiUrl): apiUrl bir Vercel service binding'inden
      // gelirse SADECE taban URL'dir (path yok); mutlak "/predict" path'i
      // her zaman base'in path'ini override ettiği için apiUrl zaten path
      // içeren eski .env.local formatıyla da (http://host:port/predict)
      // aynı sonucu üretir - geriye dönük kırılma yok.
      const upstream = await fetch(new URL("/predict", apiUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(canonicalInput),
        signal: AbortSignal.timeout(15_000),
      });
      if (!upstream.ok) {
        throw new Error(`Upstream responded ${upstream.status}`);
      }
      const data = (await upstream.json()) as Record<string, unknown>;
      // Modelin yanıt alanı esnek: price / prediction / predicted_price kabul edilir.
      const price = Number(
        data.price ?? data.prediction ?? data.predicted_price
      );
      if (!Number.isFinite(price) || price <= 0) {
        throw new Error("Upstream response missing a numeric price");
      }
      return NextResponse.json({
        price: Math.round(price),
        currency: "TRY",
        source: "model",
      });
    } catch (err) {
      console.error("[predict] upstream error:", err);
      return NextResponse.json(
        { error: "Tahmin servisine şu anda ulaşılamıyor. Lütfen tekrar deneyin." },
        { status: 502 }
      );
    }
  }

  return NextResponse.json({
    price: mockPredict(input),
    currency: "TRY",
    source: "mock",
  });
}
