"use client";

/**
 * Sahibinden.com tarzı tam-sayfa adım-adım araç seçici: Kategori › Marka ›
 * Model › Motor › Paket, her kademe tam genişlikte kendi ekranı olarak
 * sırayla açılır (küçük bir dropdown panel İÇİNDE DEĞİL — PredictionForm'un
 * "vehicle" makro-adımının TEK içeriği budur), breadcrumb ile geri dönülür.
 * Kategori (Otomobil/Arazi, SUV & Pickup/Minivan & Panelvan/Elektrikli Araç)
 * eğitim-zamanı bir alan DEĞİL, sadece Marka/Model listesini daraltan bir
 * önfiltre (bkz. lib/vehicle-category.ts) — /predict'e hiç gönderilmez, bu
 * yüzden VehicleSelection'ın bir parçası değil, bileşenin kendi local state'i.
 * Paket seçilince (veya bir kademede seçenek yoksa daha erken) onComplete()
 * çağrılır — PredictionForm bunu "vehicle" makro-adımından "details"e geçiş
 * sinyali olarak kullanır. Tek veri kaynağı: lib/validation.ts'in
 * vehicle-options.generated.ts üzerinden sunduğu getModelsForBrand /
 * getEnginesForModel / getPaketOptions.
 */
import { useMemo, useState } from "react";
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
import {
  CATEGORY_ICON,
  getBrandLabelsForCategory,
  getModelLabelsForBrandAndCategory,
  VEHICLE_CATEGORIES,
  type VehicleCategory,
} from "@/lib/vehicle-category";
import { SelectionCard } from "./SelectionCard";
import { BrandCard } from "./BrandCard";

type Level = "category" | "brand" | "model" | "engine" | "trim";

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
  /** Paket seçilince/atlanınca (veya bir kademede seçenek kalmayınca) çağrılır - "vehicle" adımının bittiği sinyali. */
  onComplete: () => void;
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
const RESET_FROM_CATEGORY = { brand: "", ...RESET_FROM_BRAND };

const LEVEL_ORDER: Level[] = ["category", "brand", "model", "engine", "trim"];
const LEVEL_LABEL: Record<Level, string> = {
  category: "Kategori",
  brand: "Marka",
  model: "Model",
  engine: "Motor",
  trim: "Paket",
};

export function VehicleSelector({ value, onChange, onComplete, brandError, modelError }: VehicleSelectorProps) {
  const [category, setCategory] = useState<VehicleCategory | null>(null);
  const [level, setLevel] = useState<Level>("category");
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  const models = useMemo(() => {
    if (!value.brand) return [];
    return category ? getModelLabelsForBrandAndCategory(value.brand, category) : [...getModelsForBrand(value.brand)];
  }, [value.brand, category]);

  const engines = useMemo(() => {
    if (!value.brand || !value.model) return [];
    const all = getEnginesForModel(value.brand, value.model);
    return category === "Elektrikli Araç" ? all.filter((e) => e.yakitTuru === "Elektrik") : all;
  }, [value.brand, value.model, category]);

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

  function levelBefore(current: Level): Level {
    const i = LEVEL_ORDER.indexOf(current);
    return LEVEL_ORDER[Math.max(i - 1, 0)];
  }

  function selectCategory(cat: VehicleCategory | null) {
    setCategory(cat);
    onChange({ ...value, ...RESET_FROM_CATEGORY });
    goToLevel("brand");
  }

  function selectBrand(brand: string) {
    onChange({ ...value, brand, ...RESET_FROM_BRAND });
    const brandModels = category ? getModelLabelsForBrandAndCategory(brand, category) : getModelsForBrand(brand);
    if (brandModels.length === 0) {
      onComplete();
      return;
    }
    goToLevel("model");
  }

  function selectModel(model: string) {
    onChange({ ...value, model, ...RESET_FROM_MODEL });
    const allEngines = getEnginesForModel(value.brand, model);
    const modelEngines = category === "Elektrikli Araç" ? allEngines.filter((e) => e.yakitTuru === "Elektrik") : allEngines;
    if (modelEngines.length === 0) {
      onComplete();
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
      onComplete();
      return;
    }
    goToLevel("trim");
  }

  function selectTrim(trim: string) {
    onChange({ ...value, trim });
    onComplete();
  }

  const normalizedQuery = query.trim().toLocaleLowerCase("tr");
  const matchesQuery = (label: string) =>
    !normalizedQuery || label.toLocaleLowerCase("tr").includes(normalizedQuery);

  const rows: Row[] = useMemo(() => {
    if (level === "category") {
      const categoryRows = VEHICLE_CATEGORIES.filter(matchesQuery).map((cat) => ({
        key: cat,
        label: cat,
        icon: CATEGORY_ICON[cat],
        selected: cat === category,
        select: () => selectCategory(cat),
      }));
      const skipRow: Row = {
        key: "__skip__",
        label: "Kategori seçmeden devam et",
        selected: category === null,
        select: () => selectCategory(null),
      };
      return [...categoryRows, skipRow];
    }
    if (level === "brand") {
      const brandLabels = category ? getBrandLabelsForCategory(category) : BRANDS;
      return brandLabels.filter(matchesQuery).map((label) => ({
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
  }, [level, query, category, models, engines, trims, value.brand, value.model, value.engineHacmiBucket, value.yakitTuru, value.trim]);

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Backspace" && query === "") {
      goToLevel(levelBefore(level));
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
  const hasError = Boolean(brandError || modelError);

  return (
    <div className="flex flex-col gap-3">
      <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">Araç Seçimi</span>
      {hasError ? (
        <p role="alert" className="text-xs text-red-400">
          {brandError || modelError}
        </p>
      ) : null}

      <div
        role="group"
        aria-label="Kategori, marka, model, motor ve paket seçimi"
        onKeyDown={handleKeyDown}
        className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02]"
      >
        <div className="flex flex-wrap items-center gap-1 border-b border-white/10 px-3 py-2.5 text-xs">
          <button
            type="button"
            aria-label="Geri"
            disabled={level === "category"}
            onClick={() => goToLevel(levelBefore(level))}
            className="rounded px-1.5 py-0.5 text-zinc-400 hover:bg-white/10 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
          >
            ‹
          </button>
          <button
            type="button"
            onClick={() => goToLevel("category")}
            disabled={level === "category"}
            className="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-white/10 hover:text-zinc-100 disabled:text-zinc-100 disabled:hover:bg-transparent"
          >
            {category ?? "Kategori"}
          </button>
          {level !== "category" ? (
            <>
              <span aria-hidden className="text-zinc-600">›</span>
              <button
                type="button"
                onClick={() => goToLevel("brand")}
                disabled={level === "brand"}
                className="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-white/10 hover:text-zinc-100 disabled:text-zinc-100 disabled:hover:bg-transparent"
              >
                {value.brand || "Marka"}
              </button>
            </>
          ) : null}
          {value.brand && (level === "model" || level === "engine" || level === "trim") ? (
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
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActiveIndex(0);
          }}
          placeholder={`${LEVEL_LABEL[level]} ara…`}
          className="w-full border-b border-white/10 bg-transparent px-3.5 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none"
        />

        <div className="max-h-[26rem] overflow-y-auto p-3">
          {rows.length === 0 ? (
            <p className="px-3 py-4 text-center text-sm text-zinc-500">Sonuç bulunamadı.</p>
          ) : level === "category" ? (
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
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
                  <SelectionCard key={row.key} label={row.label} icon={row.icon} selected={row.selected} onSelect={row.select} />
                )
              )}
            </div>
          ) : level === "brand" ? (
            <div className="grid grid-cols-3 gap-2.5 sm:grid-cols-4">
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
            <div className="flex flex-wrap gap-2.5">
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
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
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
            <div className="grid grid-cols-3 gap-2.5 sm:grid-cols-4">
              {rows.map((row) => (
                <SelectionCard key={row.key} label={row.label} selected={row.selected} onSelect={row.select} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
