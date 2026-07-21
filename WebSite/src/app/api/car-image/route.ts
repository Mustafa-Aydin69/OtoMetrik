import { NextResponse } from "next/server";
import { getVehicleImages } from "@/lib/vehicle-image/vehicle-image.service";

/**
 * POST /api/car-image
 * Tahmin formundaki marka/model/yıl/paket/renk bilgisinden, nesil (generation)
 * çıkarımı + çok kaynaklı arama + sıralama pipeline'ıyla (bkz.
 * lib/vehicle-image/vehicle-image.service.ts) en iyi eşleşen araç
 * fotoğraflarının URL listesini döner (en iyi eşleşme önce). Hiçbir aday
 * bulunamazsa boş liste döner (özellik sessizce devre dışı kalır — /api/predict'in
 * PREDICTION_API_URL yoksa mock'a düşmesiyle aynı "graceful" desen).
 */
export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ imageUrls: [] }, { status: 400 });
  }

  const b = body as Record<string, unknown>;
  const brand = typeof b.brand === "string" ? b.brand : "";
  const model = typeof b.model === "string" ? b.model : "";
  if (!brand.trim() || !model.trim()) {
    return NextResponse.json({ imageUrls: [] }, { status: 400 });
  }

  const imageUrls = await getVehicleImages({
    brand,
    model,
    year: typeof b.year === "number" && Number.isFinite(b.year) ? b.year : null,
    trim: typeof b.trim === "string" ? b.trim : null,
    color: typeof b.color === "string" ? b.color : null,
  });

  return NextResponse.json({ imageUrls });
}
