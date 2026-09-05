"use client";

/**
 * VehicleSelector'ın Model/Motor/Paket kademelerinin ortak, tek amaçlı kart
 * bileşeni. "tile" (Model/Paket - kare ızgara kartı) ve "pill" (Motor -
 * yuvarlak hatlı satır) iki görsel varyant paylaşır; seçili/hover/devre dışı
 * durumları tek yerde tanımlanır.
 */
interface SelectionCardProps {
  label: string;
  caption?: string;
  icon?: string;
  selected: boolean;
  disabled?: boolean;
  disabledReason?: string;
  variant?: "tile" | "pill";
  onSelect: () => void;
}

export function SelectionCard({
  label,
  caption,
  icon,
  selected,
  disabled,
  disabledReason,
  variant = "tile",
  onSelect,
}: SelectionCardProps) {
  const base =
    "relative flex items-center gap-2.5 border transition-colors focus-visible:outline-2 " +
    "focus-visible:outline-accent focus-visible:outline-offset-2";
  const shape =
    variant === "pill"
      ? "rounded-full px-4 py-2.5 flex-row"
      : "flex-col items-center gap-2 rounded-xl px-3 py-4 text-center";
  const tone = selected
    ? "border-accent/70 bg-accent/10 shadow-[0_0_0_1px_rgba(232,179,60,0.35)]"
    : disabled
      ? "cursor-not-allowed border-white/5 bg-white/[0.02] opacity-40"
      : "cursor-pointer border-white/10 bg-white/[0.04] hover:-translate-y-0.5 hover:border-white/20 hover:bg-white/[0.07]";

  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={selected}
      onClick={onSelect}
      className={`${base} ${shape} ${tone}`}
    >
      {selected ? (
        <span className="absolute right-2 top-2 grid size-4 place-items-center rounded-full bg-accent text-[10px] font-bold text-zinc-950">
          ✓
        </span>
      ) : null}
      {icon ? (
        <span className="grid size-6 shrink-0 place-items-center rounded-full bg-white/[0.06] text-xs">
          {icon}
        </span>
      ) : null}
      <span className={variant === "pill" ? "flex flex-col items-start gap-0.5" : "flex flex-col gap-1"}>
        <span className="text-sm font-medium text-zinc-100">{label}</span>
        {caption ? (
          <span className="text-[11px] text-zinc-500">{caption}</span>
        ) : null}
        {disabled && disabledReason ? (
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            🔒 {disabledReason}
          </span>
        ) : null}
      </span>
    </button>
  );
}
