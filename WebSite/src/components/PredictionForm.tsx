"use client";

/**
 * Fiyat tahmin formu: alan doğrulaması, loading/hata/tekrar deneme akışı ve
 * başarılı sonuçta PredictionResult kartı.
 */
import { useRef, useState } from "react";
import {
  BODY_TYPES,
  BRANDS,
  COLORS,
  CURRENT_YEAR,
  FUEL_TYPES,
  MIN_YEAR,
  TRANSMISSIONS,
  validatePrediction,
  type FieldErrors,
  type PredictionInput,
} from "@/lib/validation";
import { requestPrediction, type PredictionResponse } from "@/lib/prediction";
import { NumberField, SelectField, TextField, YesNoField } from "./fields";
import { PredictionResult } from "./PredictionResult";

const MODEL_SUGGESTIONS: Record<string, readonly string[]> = {
  Fiat: ["Egea", "Doblo", "Fiorino", "Panda", "500"],
  Ford: ["Focus", "Fiesta", "Kuga", "Puma", "Transit", "Mustang"],
  Volkswagen: ["Golf", "Polo", "Passat", "Tiguan", "T-Roc", "Caddy"],
  Renault: ["Clio", "Megane", "Symbol", "Captur", "Taliant"],
  Peugeot: ["208", "308", "2008", "3008", "508"],
  Toyota: ["Corolla", "Yaris", "C-HR", "RAV4"],
  Hyundai: ["i20", "i10", "Tucson", "Bayon", "Elantra"],
  Opel: ["Corsa", "Astra", "Crossland", "Mokka"],
  BMW: ["3 Serisi", "5 Serisi", "1 Serisi", "X1", "X3"],
  "Mercedes-Benz": ["C Serisi", "E Serisi", "A Serisi", "GLA", "CLA"],
  Audi: ["A3", "A4", "A6", "Q2", "Q3"],
};

const YEARS = Array.from(
  { length: CURRENT_YEAR + 1 - MIN_YEAR + 1 },
  (_, i) => CURRENT_YEAR + 1 - i
);

interface FormState {
  brand: string;
  model: string;
  year: string;
  mileage: string;
  fuelType: string;
  transmission: string;
  bodyType: string;
  color: string;
  engineDisplacement: string;
  enginePower: string;
  trim: string;
  replacedPartsCount: string;
  paintedPartsCount: string;
  heavyDamage: boolean;
}

const INITIAL: FormState = {
  brand: "",
  model: "",
  year: "",
  mileage: "",
  fuelType: "",
  transmission: "",
  bodyType: "",
  color: "",
  engineDisplacement: "",
  enginePower: "",
  trim: "",
  replacedPartsCount: "0",
  paintedPartsCount: "0",
  heavyDamage: false,
};

function toInput(f: FormState): PredictionInput {
  return {
    brand: f.brand.trim(),
    model: f.model.trim(),
    year: Number(f.year),
    mileage: Number(f.mileage),
    fuelType: f.fuelType,
    transmission: f.transmission,
    bodyType: f.bodyType,
    color: f.color,
    engineDisplacement:
      f.fuelType === "Elektrik" && f.engineDisplacement === ""
        ? 0
        : Number(f.engineDisplacement),
    enginePower: Number(f.enginePower),
    trim: f.trim.trim(),
    replacedPartsCount: Number(f.replacedPartsCount),
    paintedPartsCount: Number(f.paintedPartsCount),
    heavyDamage: f.heavyDamage,
  };
}

type Status = "idle" | "loading" | "error" | "success";

export function PredictionForm() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [status, setStatus] = useState<Status>("idle");
  const [apiError, setApiError] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  function set<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
    // Alan düzeltildiğinde o alanın hatasını temizle.
    if (key in errors) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key as keyof FieldErrors];
        return next;
      });
    }
  }

  async function submit() {
    const input = toInput(form);
    const fieldErrors = validatePrediction(input);
    setErrors(fieldErrors);
    if (Object.keys(fieldErrors).length > 0) {
      setStatus("idle");
      return;
    }

    setStatus("loading");
    setApiError("");
    setResult(null);
    try {
      const res = await requestPrediction(input);
      setResult(res);
      setStatus("success");
      requestAnimationFrame(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Beklenmeyen bir hata oluştu.");
      setStatus("error");
    }
  }

  const modelSuggestions = MODEL_SUGGESTIONS[form.brand] ?? [];

  return (
    <div className="mx-auto w-full max-w-3xl">
      <form
        noValidate
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        className="rounded-2xl border border-white/10 bg-white/[0.04] p-6 shadow-2xl shadow-black/40 backdrop-blur-xl sm:p-8"
      >
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          <SelectField
            id="brand"
            label="Marka"
            value={form.brand}
            onChange={(v) => set("brand", v)}
            options={BRANDS}
            error={errors.brand}
            placeholder="Marka seçin"
          />
          <TextField
            id="model"
            label="Model"
            value={form.model}
            onChange={(v) => set("model", v)}
            error={errors.model}
            placeholder="örn. Focus"
            suggestions={modelSuggestions}
          />
          <SelectField
            id="year"
            label="Yıl"
            value={form.year}
            onChange={(v) => set("year", v)}
            options={YEARS}
            error={errors.year}
            placeholder="Yıl seçin"
          />
          <NumberField
            id="mileage"
            label="Kilometre"
            value={form.mileage}
            onChange={(v) => set("mileage", v)}
            error={errors.mileage}
            placeholder="örn. 85000"
            min={0}
            suffix="km"
          />
          <SelectField
            id="fuelType"
            label="Yakıt Türü"
            value={form.fuelType}
            onChange={(v) => set("fuelType", v)}
            options={FUEL_TYPES}
            error={errors.fuelType}
            placeholder="Yakıt türü seçin"
          />
          <SelectField
            id="transmission"
            label="Vites"
            value={form.transmission}
            onChange={(v) => set("transmission", v)}
            options={TRANSMISSIONS}
            error={errors.transmission}
            placeholder="Vites seçin"
          />
          <SelectField
            id="bodyType"
            label="Kasa Tipi"
            value={form.bodyType}
            onChange={(v) => set("bodyType", v)}
            options={BODY_TYPES}
            error={errors.bodyType}
            placeholder="Kasa tipi seçin"
          />
          <SelectField
            id="color"
            label="Renk"
            value={form.color}
            onChange={(v) => set("color", v)}
            options={COLORS}
            error={errors.color}
            placeholder="Renk seçin"
          />
          <NumberField
            id="engineDisplacement"
            label="Motor Hacmi"
            value={form.engineDisplacement}
            onChange={(v) => set("engineDisplacement", v)}
            error={errors.engineDisplacement}
            placeholder={
              form.fuelType === "Elektrik"
                ? "Elektrikli — boş bırakılabilir"
                : "örn. 1600"
            }
            min={0}
            suffix="cc"
          />
          <NumberField
            id="enginePower"
            label="Motor Gücü"
            value={form.enginePower}
            onChange={(v) => set("enginePower", v)}
            error={errors.enginePower}
            placeholder="örn. 110"
            min={1}
            suffix="HP"
          />
          <TextField
            id="trim"
            label="Paket"
            value={form.trim}
            onChange={(v) => set("trim", v)}
            error={errors.trim}
            placeholder="örn. Titanium (opsiyonel)"
          />
          <YesNoField
            id="heavyDamage"
            label="Ağır Hasarlı"
            value={form.heavyDamage}
            onChange={(v) => set("heavyDamage", v)}
          />
          <NumberField
            id="replacedPartsCount"
            label="Değişen Sayısı"
            value={form.replacedPartsCount}
            onChange={(v) => set("replacedPartsCount", v)}
            error={errors.replacedPartsCount}
            min={0}
            max={13}
            step={1}
          />
          <NumberField
            id="paintedPartsCount"
            label="Boyalı Sayısı"
            value={form.paintedPartsCount}
            onChange={(v) => set("paintedPartsCount", v)}
            error={errors.paintedPartsCount}
            min={0}
            max={13}
            step={1}
          />
        </div>

        {status === "error" ? (
          <div
            role="alert"
            className="mt-6 flex flex-col items-start gap-3 rounded-lg border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-300 sm:flex-row sm:items-center sm:justify-between"
          >
            <span>{apiError}</span>
            <button
              type="button"
              onClick={() => void submit()}
              className="shrink-0 rounded-md border border-red-300/30 px-3 py-1.5 text-xs font-medium text-red-200 transition-colors hover:bg-red-400/10"
            >
              Tekrar Dene
            </button>
          </div>
        ) : null}

        <button
          type="submit"
          disabled={status === "loading"}
          className="mt-8 flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-50 px-6 py-3.5 text-sm font-semibold text-zinc-950 transition-all hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status === "loading" ? (
            <>
              <span
                aria-hidden
                className="size-4 animate-spin rounded-full border-2 border-zinc-400 border-t-zinc-900"
              />
              Hesaplanıyor…
            </>
          ) : (
            "Değerini Öğren"
          )}
        </button>
      </form>

      <div ref={resultRef} className="mt-8" aria-live="polite">
        {status === "success" && result ? (
          <PredictionResult price={result.price} source={result.source} />
        ) : null}
      </div>
    </div>
  );
}
