"use client";

/**
 * Sahibinden.com tarzı kademeli araç seçici: Marka → Model → Paket, aynı
 * panel içinde sırayla açılır, breadcrumb ile geri dönülür. Serbest metin
 * girişini (eski TextField + datalist) yerine bilinen kümeye kısıtlanmış
 * bir seçim akışıyla değiştirir — bkz. lib/validation.ts getModelsForBrand /
 * getPaketSuggestions (ikisi de eğitim verisinden türetilmiş tek kaynak).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { BRANDS, getModelsForBrand, getPaketSuggestions } from "@/lib/validation";
import { inputBase } from "./fields";

type Level = "brand" | "model" | "trim";

export interface VehicleSelection {
  brand: string;
  model: string;
  trim: string;
}

interface VehiclePickerProps {
  value: VehicleSelection;
  onChange: (next: VehicleSelection) => void;
  brandError?: string;
  modelError?: string;
}

interface NavItem {
  label: string;
  selected: boolean;
  select: () => void;
}

export function VehiclePicker({ value, onChange, brandError, modelError }: VehiclePickerProps) {
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
  const trims = useMemo(
    () => (value.brand && value.model ? getPaketSuggestions(value.brand, value.model) : []),
    [value.brand, value.model]
  );

  function selectBrand(brand: string) {
    onChange({ brand, model: "", trim: "" });
    if (getModelsForBrand(brand).length === 0) {
      setOpen(false);
      return;
    }
    goToLevel("model");
  }

  function selectModel(model: string) {
    onChange({ brand: value.brand, model, trim: "" });
    if (getPaketSuggestions(value.brand, model).length === 0) {
      setOpen(false);
      return;
    }
    goToLevel("trim");
  }

  function selectTrim(trim: string) {
    onChange({ brand: value.brand, model: value.model, trim });
    setOpen(false);
  }

  function goToLevel(target: Level) {
    setLevel(target);
    setQuery("");
    setActiveIndex(0);
  }

  const items = level === "brand" ? BRANDS : level === "model" ? models : trims;
  const normalizedQuery = query.trim().toLocaleLowerCase("tr");
  const filtered = normalizedQuery
    ? items.filter((item) => item.toLocaleLowerCase("tr").includes(normalizedQuery))
    : items;

  const navItems: NavItem[] = useMemo(() => {
    const rows: NavItem[] =
      level === "trim"
        ? [{ label: "Paket belirtmeden devam et", selected: value.trim === "", select: () => selectTrim("") }]
        : [];
    for (const item of filtered) {
      rows.push({
        label: item,
        selected:
          (level === "brand" && item === value.brand) ||
          (level === "model" && item === value.model) ||
          (level === "trim" && item === value.trim),
        select: () =>
          level === "brand" ? selectBrand(item) : level === "model" ? selectModel(item) : selectTrim(item),
      });
    }
    return rows;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [level, filtered, value.brand, value.model, value.trim]);

  // Panel açılınca / seviye değişince arama kutusuna odaklan. Arama+seçim
  // indeksinin sıfırlanması burada DEĞİL, seviyeyi değiştiren her olay
  // handler'ında (goToLevel, openPanel) senkron yapılır - effect içinde
  // setState kademeli render'lara yol açar (bkz. react-hooks/set-state-in-effect).
  useEffect(() => {
    if (!open) return;
    const raf = requestAnimationFrame(() => searchRef.current?.focus());
    return () => cancelAnimationFrame(raf);
  }, [open, level]);

  // Panelin dışına tıklanınca kapat.
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
    const target: Level = !value.brand ? "brand" : !value.model ? "model" : "trim";
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
      if (level === "trim") goToLevel("model");
      else if (level === "model") goToLevel("brand");
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, navItems.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      navItems[activeIndex]?.select();
    }
  }

  const triggerLabel = value.brand
    ? [value.brand, value.model, value.trim].filter(Boolean).join(" › ")
    : "Araç seçin";
  const hasError = Boolean(brandError || modelError);

  return (
    <div className="flex flex-col gap-1.5 sm:col-span-2" ref={containerRef}>
      <span id="vehicle-picker-label" className="text-xs font-medium uppercase tracking-wider text-zinc-400">
        Marka / Model / Paket
      </span>
      <button
        type="button"
        className={`${inputBase} flex items-center justify-between gap-2 text-left ${
          value.brand ? "text-zinc-100" : "text-zinc-500"
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby="vehicle-picker-label"
        aria-describedby={hasError ? "vehicle-picker-error" : undefined}
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
        <p id="vehicle-picker-error" role="alert" className="text-xs text-red-400">
          {brandError || modelError}
        </p>
      ) : null}

      {open ? (
        <div
          role="listbox"
          aria-label="Marka, model ve paket seçimi"
          onKeyDown={handleKeyDown}
          className="relative z-10 mt-1 overflow-hidden rounded-lg border border-white/10 bg-zinc-950/95 shadow-2xl shadow-black/50 backdrop-blur-xl"
        >
          <div className="flex items-center gap-1 border-b border-white/10 px-2 py-2 text-xs">
            <button
              type="button"
              aria-label="Geri"
              disabled={level === "brand"}
              onClick={() => goToLevel(level === "trim" ? "model" : "brand")}
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
                <span aria-hidden className="text-zinc-600">
                  ›
                </span>
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
            {level === "trim" ? (
              <>
                <span aria-hidden className="text-zinc-600">
                  ›
                </span>
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
            placeholder={level === "brand" ? "Marka ara…" : level === "model" ? "Model ara…" : "Paket ara…"}
            className="w-full border-b border-white/10 bg-transparent px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none"
          />
          <ul className="max-h-72 overflow-y-auto py-1">
            {navItems.length === 0 ? (
              <li className="px-3 py-4 text-center text-sm text-zinc-500">Sonuç bulunamadı.</li>
            ) : (
              navItems.map((item, i) => (
                <li key={item.label}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={item.selected}
                    onMouseEnter={() => setActiveIndex(i)}
                    onClick={item.select}
                    className={`block w-full px-3 py-2 text-left text-sm ${
                      i === activeIndex || item.selected
                        ? "bg-sky-500/20 text-sky-200"
                        : "text-zinc-300 hover:bg-white/5"
                    }`}
                  >
                    {item.label}
                  </button>
                </li>
              ))
            )}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
