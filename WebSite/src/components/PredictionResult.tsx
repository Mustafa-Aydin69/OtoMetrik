"use client";

/**
 * Başarılı tahmin sonrası gösterilen premium sonuç kartı.
 * Mount olduğunda yumuşak bir fade/slide animasyonuyla görünür.
 */
import { useEffect, useState } from "react";
import { formatTRY } from "@/lib/prediction";

export function PredictionResult({
  price,
  source,
  imageUrls = [],
  carLabel = "Araç",
}: {
  price: number;
  source: "model" | "mock";
  imageUrls?: string[];
  carLabel?: string;
}) {
  const [shown, setShown] = useState(false);
  const [urlIndex, setUrlIndex] = useState(0);
  const [trackedImageUrls, setTrackedImageUrls] = useState(imageUrls);

  useEffect(() => {
    const t = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(t);
  }, []);

  // Yeni bir tahminde (farklı marka/model/renk) kademeyi baştan başlat.
  // Effect yerine render sırasında state ayarlama (React'ın önerdiği desen,
  // bkz. https://react.dev/learn/you-might-not-need-an-effect) — cascading
  // effect render'ı yerine tek seferde re-render tetikler.
  if (imageUrls !== trackedImageUrls) {
    setTrackedImageUrls(imageUrls);
    setUrlIndex(0);
  }

  const currentImageUrl = urlIndex < imageUrls.length ? imageUrls[urlIndex] : null;

  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/[0.06] p-8 text-center shadow-2xl shadow-black/40 backdrop-blur-xl transition-all duration-700 ease-out ${
        shown ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"
      }`}
      role="status"
      aria-live="polite"
    >
      {currentImageUrl ? (
        // eslint-disable-next-line @next/next/no-img-element -- harici imagin.studio CDN, next/image domain izni gerektirmesin diye düz <img>.
        <img
          key={currentImageUrl}
          src={currentImageUrl}
          alt={`${carLabel} görseli`}
          className="mx-auto mb-6 h-48 w-full max-w-sm object-contain sm:h-56"
          onError={() => setUrlIndex((i) => i + 1)}
        />
      ) : null}
      <p className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-400">
        Tahmini Araç Değeri
      </p>
      <p className="mt-4 text-5xl font-semibold tracking-tight text-white sm:text-6xl">
        {formatTRY(price)}
      </p>
      <p className="mx-auto mt-5 max-w-md text-sm leading-relaxed text-zinc-400">
        Bu değer, araç bilgilerinize dayanan istatistiksel bir tahmindir; nihai
        satış fiyatı piyasa koşullarına ve aracın gerçek durumuna göre
        değişebilir.
        {source === "mock" ? (
          <span className="mt-2 block text-xs text-zinc-500">
            (Geliştirme modu — demo tahmini)
          </span>
        ) : null}
      </p>
    </div>
  );
}
