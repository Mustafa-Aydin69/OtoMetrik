"use client";

/**
 * Sahibinden.com tarzı kart tabanlı kademeli araç seçici: Marka › Model ›
 * Motor › Paket, aynı panel içinde sırayla açılır, breadcrumb ile geri
 * dönülür. VehiclePicker.tsx'in (liste/panel tabanlı 3 kademeli) yerine
 * geçer — kartlara (marka için gerçek logo, bkz. BrandCard) ve araya bir
 * Motor kademesine (motor hacmi + yakıt türü) sahip. Tek veri kaynağı:
 * lib/validation.ts'in vehicle-options.generated.ts üzerinden sunduğu
 * getModelsForBrand / getEnginesForModel / getPaketOptions.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import {
  BRANDS,
  canonicalToLabel,
  getEnginesForModel,
  getModelsForBrand,
  getPaketOptions,
  labelToCanonical,
  type EngineOption,
} from "@/lib/validation";
import { inputBase } from "./fields";
import { SelectionCard } from "./SelectionCard";
import { BrandCard } from "./BrandCard";

type Level = "brand" | "model" | "engine" | "trim";

export interface VehicleSelection {
  brand: string;
  model: string;
  /** Motor kademesinde seçilen kova (görüntü/gruplama, örn. 1600) - /predict'e GÖNDERİLMEZ. */
  engineHacmiBucket: number | null;
  /** O kovadaki en sık görülen gerçek cc değeri - engineDisplacement olarak GÖNDERİLİR. */
  engineExactCc: number | null;
  /** Kanonik yakıt türü (örn. "LPG & Benzin") - fuelType'a canonicalToLabel ile çevrilir. */
  yakitTuru: string;
  trim: string;
}

interface VehicleSelectorProps {
  value: VehicleSelection;
  onChange: (next: VehicleSelection) => void;
  brandError?: string;
  modelError?: string;
}

interface Row {
  key: string;
  label: string;
  icon?: string;
  selected: boolean;
  select: () => void;
}

function fuelIcon(yakitTuru: string): string {
  if (yakitTuru === "Dizel") return "💧";
  if (yakitTuru === "Elektrik") return "🔌";
  if (yakitTuru === "Hibrit") return "🔋";
  return "⚡";
}

function engineLabel(engine: Pick<EngineOption, "hacmiBucket" | "yakitTuru">): string {
  const displacement = (engine.hacmiBucket / 1000).toFixed(1);
  return `${displacement} · ${canonicalToLabel("fuelType", engine.yakitTuru)}`;
}

const RESET_FROM_BRAND = { model: "", engineHacmiBucket: null, engineExactCc: null, yakitTuru: "", trim: "" };
const RESET_FROM_MODEL = { engineHacmiBucket: null, engineExactCc: null, yakitTuru: "", trim: "" };

export function VehicleSelector({ value, onChange, brandError, modelError }: VehicleSelectorProps) {
  const [open, setOpen] = useState(false);
  const [level, setLevel] = useState<Level>("brand");
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const models = useMemo(
    () => (value.brand ? getModelsForBrand(value.brand) : []),
    [value.brand]
  );
  const engines = useMemo(
    () => (value.brand && value.model ? getEnginesForModel(value.brand, value.model) : []),
    [value.brand, value.model]
  );
  const trims = useMemo(
    () =>
      value.brand && value.model && value.engineHacmiBucket !== null
        ? getPaketOptions(value.brand, value.model, value.engineHacmiBucket, value.yakitTuru)
        : [],
    [value.brand, value.model, value.engineHacmiBucket, value.yakitTuru]
  );

  function goToLevel(target: Level) {
    setLevel(target);
    setQuery("");
    setActiveIndex(0);
  }

  function selectBrand(brand: string) {
    onChange({ ...value, brand, ...RESET_FROM_BRAND });
    if (getModelsForBrand(brand).length === 0) {
      setOpen(false);
      return;
    }
    goToLevel("model");
  }

  function selectModel(model: string) {
    onChange({ ...value, model, ...RESET_FROM_MODEL });
    if (getEnginesForModel(value.brand, model).length === 0) {
      setOpen(false);
      return;
    }
    goToLevel("engine");
  }

  function selectEngine(engine: EngineOption) {
    onChange({
      ...value,
      engineHacmiBucket: engine.hacmiBucket,
      engineExactCc: engine.exactCc,
      yakitTuru: engine.yakitTuru,
      trim: "",
    });
    if (getPaketOptions(value.brand, value.model, engine.hacmiBucket, engine.yakitTuru).length === 0) {
      setOpen(false);
      return;
    }
    goToLevel("trim");
  }

  function selectTrim(trim: string) {
    onChange({ ...value, trim });
    setOpen(false);
  }

  const normalizedQuery = query.trim().toLocaleLowerCase("tr");
  const matchesQuery = (label: string) =>
    !normalizedQuery || label.toLocaleLowerCase("tr").includes(normalizedQuery);

  const rows: Row[] = useMemo(() => {
    if (level === "brand") {
      return BRANDS.filter(matchesQuery).map((label) => ({
        key: label,
        label,
        selected: label === value.brand,
        select: () => selectBrand(label),
      }));
    }
    if (level === "model") {
      return models.filter(matchesQuery).map((label) => ({
        key: label,
        label,
        selected: label === value.model,
        select: () => selectModel(label),
      }));
    }
    if (level === "engine") {
      return engines
        .map((engine) => ({ engine, label: engineLabel(engine) }))
        .filter((row) => matchesQuery(row.label))
        .map(({ engine, label }) => ({
          key: `${engine.hacmiBucket}|${engine.yakitTuru}`,
          label,
          icon: fuelIcon(engine.yakitTuru),
          selected: engine.hacmiBucket === value.engineHacmiBucket && engine.yakitTuru === value.yakitTuru,
          select: () => selectEngine(engine),
        }));
    }
    const skipRow: Row = {
      key: "__skip__",
      label: "Paket belirtmeden devam et",
      selected: value.trim === "",
      select: () => selectTrim(""),
    };
    const paketRows = trims.filter(matchesQuery).map((label) => ({
      key: label,
      label,
      selected: label === value.trim,
      select: () => selectTrim(label),
    }));
    return [skipRow, ...paketRows];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [level, query, models, engines, trims, value.brand, value.model, value.engineHacmiBucket, value.yakitTuru, value.trim]);

  useEffect(() => {
    if (!open) return;
    const raf = requestAnimationFrame(() => searchRef.current?.focus());
    return () => cancelAnimationFrame(raf);
  }, [open, level]);

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(e: PointerEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [open]);

  function openPanel() {
    const target: Level = !value.brand
      ? "brand"
      : !value.model
        ? "model"
        : value.engineHacmiBucket === null
          ? "engine"
          : "trim";
    setLevel(target);
    setQuery("");
    setActiveIndex(0);
    setOpen(true);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (e.key === "Backspace" && query === "") {
      if (level === "trim") goToLevel("engine");
      else if (level === "engine") goToLevel("model");
      else if (level === "model") goToLevel("brand");
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, rows.length - 1));
      return;
    }
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      rows[activeIndex]?.select();
    }
  }

  const currentEngineLabel =
    value.engineHacmiBucket !== null ? engineLabel({ hacmiBucket: value.engineHacmiBucket, yakitTuru: value.yakitTuru }) : "";
  const triggerLabel = value.brand
    ? [value.brand, value.model, currentEngineLabel, value.trim].filter(Boolean).join(" › ")
    : "Araç seçin";
  const hasError = Boolean(brandError || modelError);

  const levelLabel: Record<Level, string> = {
    brand: "Marka",
    model: "Model",
    engine: "Motor",
    trim: "Paket",
  };

  return (
    <div className="flex flex-col gap-1.5 sm:col-span-2" ref={containerRef}>
      <span id="vehicle-selector-label" className="text-xs font-medium uppercase tracking-wider text-zinc-400">
        Marka / Model / Motor / Paket
      </span>
      <button
        type="button"
        className={`${inputBase} flex items-center justify-between gap-2 text-left ${
          value.brand ? "text-zinc-100" : "text-zinc-500"
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby="vehicle-selector-label"
        aria-describedby={hasError ? "vehicle-selector-error" : undefined}
        onClick={() => (open ? setOpen(false) : openPanel())}
      >
        <span className="truncate">{triggerLabel}</span>
        <svg
          aria-hidden
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#a1a1aa"
          strokeWidth="2"
          className="shrink-0"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {hasError ? (
        <p id="vehicle-selector-error" role="alert" className="text-xs text-red-400">
          {brandError || modelError}
        </p>
      ) : null}

      {open ? (
        <div
          role="listbox"
          aria-label="Marka, model, motor ve paket seçimi"
          onKeyDown={handleKeyDown}
          className="relative z-10 mt-1 overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl shadow-black/50 backdrop-blur-xl"
        >
          <div className="flex items-center gap-1 border-b border-white/10 px-2 py-2 text-xs">
            <button
              type="button"
              aria-label="Geri"
              disabled={level === "brand"}
              onClick={() => goToLevel(level === "trim" ? "engine" : level === "engine" ? "model" : "brand")}
              className="rounded px-1.5 py-0.5 text-zinc-400 hover:bg-white/10 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
            >
              ‹
            </button>
            <button
              type="button"
              onClick={() => goToLevel("brand")}
              disabled={level === "brand"}
              className="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-white/10 hover:text-zinc-100 disabled:text-zinc-100 disabled:hover:bg-transparent"
            >
              {value.brand || "Marka"}
            </button>
            {value.brand && level !== "brand" ? (
              <>
                <span aria-hidden className="text-zinc-600">›</span>
                <button
                  type="button"
                  onClick={() => goToLevel("model")}
                  disabled={level === "model"}
                  className="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-white/10 hover:text-zinc-100 disabled:text-zinc-100 disabled:hover:bg-transparent"
                >
                  {value.model || "Model"}
                </button>
              </>
            ) : null}
            {value.model && (level === "engine" || level === "trim") ? (
              <>
                <span aria-hidden className="text-zinc-600">›</span>
                <button
                  type="button"
                  onClick={() => goToLevel("engine")}
                  disabled={level === "engine"}
                  className="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-white/10 hover:text-zinc-100 disabled:text-zinc-100 disabled:hover:bg-transparent"
                >
                  {currentEngineLabel || "Motor"}
                </button>
              </>
            ) : null}
            {level === "trim" ? (
              <>
                <span aria-hidden className="text-zinc-600">›</span>
                <span className="px-1.5 py-0.5 text-zinc-500">Paket</span>
              </>
            ) : null}
          </div>

          <input
            ref={searchRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            placeholder={`${levelLabel[level]} ara…`}
            className="w-full border-b border-white/10 bg-transparent px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none"
          />

          <div className="max-h-80 overflow-y-auto p-2.5">
            {rows.length === 0 ? (
              <p className="px-3 py-4 text-center text-sm text-zinc-500">Sonuç bulunamadı.</p>
            ) : level === "brand" ? (
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                {rows.map((row) => (
                  <BrandCard
                    key={row.key}
                    canonicalBrand={labelToCanonical("brand", row.label)}
                    label={row.label}
                    selected={row.selected}
                    onSelect={row.select}
                  />
                ))}
              </div>
            ) : level === "engine" ? (
              <div className="flex flex-wrap gap-2">
                {rows.map((row) => (
                  <SelectionCard
                    key={row.key}
                    variant="pill"
                    label={row.label}
                    icon={row.icon}
                    selected={row.selected}
                    onSelect={row.select}
                  />
                ))}
              </div>
            ) : level === "trim" ? (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {rows.map((row) =>
                  row.key === "__skip__" ? (
                    <button
                      key={row.key}
                      type="button"
                      onClick={row.select}
                      className="col-span-full rounded-xl border border-dashed border-white/15 px-3 py-2.5 text-center text-xs text-zinc-500 transition-colors hover:border-white/30 hover:text-zinc-300"
                    >
                      {row.label}
                    </button>
                  ) : (
                    <SelectionCard key={row.key} label={row.label} selected={row.selected} onSelect={row.select} />
                  )
                )}
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                {rows.map((row) => (
                  <SelectionCard key={row.key} label={row.label} selected={row.selected} onSelect={row.select} />
                ))}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
