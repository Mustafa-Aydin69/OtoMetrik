"use client";

/**
 * Marka kademesinin kartı — SelectionCard'dan ayrı, çünkü logo çözümleme
 * mantığı var: brand-logo-map.ts'te slug bulunursa gerçek marka logosu
 * (WebSite/public/brand-logos/{slug}.svg), bulunamazsa baş harflerden oluşan
 * metin rozetine düşer (bkz. plan: "logolar tüm markalarda yok, zarif
 * text-only karta düşülmeli").
 */
import { getBrandLogoSlug } from "@/lib/brand-logo-map";

interface BrandCardProps {
  canonicalBrand: string;
  label: string;
  selected: boolean;
  onSelect: () => void;
}

function initialsOf(label: string): string {
  const words = label.split(/[\s-]+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

export function BrandCard({ canonicalBrand, label, selected, onSelect }: BrandCardProps) {
  const slug = getBrandLogoSlug(canonicalBrand);

  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={`relative flex flex-col items-center gap-2 rounded-xl border px-3 py-4 text-center transition-colors focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-2 ${
        selected
          ? "border-accent/70 bg-accent/10 shadow-[0_0_0_1px_rgba(232,179,60,0.35)]"
          : "cursor-pointer border-white/10 bg-white/[0.04] hover:-translate-y-0.5 hover:border-white/20 hover:bg-white/[0.07]"
      }`}
    >
      {selected ? (
        <span className="absolute right-2 top-2 grid size-4 place-items-center rounded-full bg-accent text-[10px] font-bold text-zinc-950">
          ✓
        </span>
      ) : null}
      <span className="grid size-9 shrink-0 place-items-center overflow-hidden rounded-full bg-zinc-100 shadow-inner">
        {slug ? (
          // eslint-disable-next-line @next/next/no-img-element -- statik marka logoları, Next Image optimizasyonuna gerek yok
          <img src={`/brand-logos/${slug}.svg`} alt="" aria-hidden className="size-5 object-contain" />
        ) : (
          <span className="text-[11px] font-bold text-zinc-800">{initialsOf(label)}</span>
        )}
      </span>
      <span className="text-sm font-medium text-zinc-100">{label}</span>
    </button>
  );
}
