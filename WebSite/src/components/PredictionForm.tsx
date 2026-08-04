"use client";

/**
 * Fiyat tahmin formu: alan doğrulaması, loading/hata/tekrar deneme akışı ve
 * başarılı sonuçta PredictionResult kartı.
 */
import { useMemo, useRef, useState } from "react";
import {
  BODY_TYPES,
  canonicalToLabel,
  COLORS,
  CURRENT_YEAR,
  FUEL_TYPES,
  getHpOptions,
  MIN_YEAR,
  TRANSMISSIONS,
  validatePrediction,
  type FieldErrors,
  type PredictionInput,
} from "@/lib/validation";
import { requestPrediction, type PredictionResponse } from "@/lib/prediction";
import { requestCarImageUrls } from "@/lib/car-photo";
import { FormSection } from "./FormSection";
import { LockedField, NumberField, SelectField, YesNoField } from "./fields";
import { PredictionResult } from "./PredictionResult";
import { VehicleSelector, type VehicleSelection } from "./VehicleSelector";

const YEARS = Array.from(
  { length: CURRENT_YEAR + 1 - MIN_YEAR + 1 },
  (_, i) => CURRENT_YEAR + 1 - i
);

interface FormState {
  brand: string;
  model: string;
  /** VehicleSelector'ın Motor kademesinden - LockedField'ların "kilitli mi?" kararı buna bakar. */
  engineHacmiBucket: number | null;
  engineExactCc: number | null;
  /** Kanonik yakıt türü (örn. "LPG & Benzin") - fuelType (etiket) bundan türetilir. */
  yakitTuru: string;
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
  engineHacmiBucket: null,
  engineExactCc: null,
  yakitTuru: "",
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
  const [carImageUrls, setCarImageUrls] = useState<string[]>([]);
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

  function setVehicle(next: VehicleSelection) {
    // Marka/model/motor/paket VehicleSelector içinde birlikte değişir (marka
    // değişince model+motor+paket sıfırlanır) - tek atomik güncelleme
    // gerekir. Motor Hacmi/Yakıt Türü doğrudan seçimden türetilir (bkz.
    // LockedField); Motor Gücü'nün TEK geçerli değeri varsa o da otomatik
    // doldurulur, birden fazla varsa kullanıcı aşağıdaki SelectField'dan
    // seçene kadar boş kalır (bkz. render'daki hpOptions).
    const fuelLabel = next.yakitTuru ? canonicalToLabel("fuelType", next.yakitTuru) : "";
    const hpOptions =
      next.brand && next.model && next.engineHacmiBucket !== null
        ? getHpOptions(next.brand, next.model, next.engineHacmiBucket, next.yakitTuru)
        : [];

    setForm((prev) => ({
      ...prev,
      brand: next.brand,
      model: next.model,
      trim: next.trim,
      engineHacmiBucket: next.engineHacmiBucket,
      engineExactCc: next.engineExactCc,
      yakitTuru: next.yakitTuru,
      fuelType: fuelLabel,
      engineDisplacement: next.engineExactCc !== null ? String(next.engineExactCc) : "",
      enginePower: hpOptions.length === 1 ? String(hpOptions[0]) : "",
    }));
    setErrors((prev) => {
      const rest = { ...prev };
      delete rest.brand;
      delete rest.model;
      delete rest.fuelType;
      delete rest.engineDisplacement;
      delete rest.enginePower;
      return rest;
    });
  }

  // Motor Gücü'nün bu marka+model+motor kombinasyonu için kaç geçerli değeri
  // var: 0 -> serbest NumberField (araç/motor henüz seçilmedi VEYA bu
  // kombinasyon için veri yok), 1 -> LockedField (setVehicle zaten doldurdu),
  // >1 -> SelectField (kullanıcı seçmeli, bkz. plan "HP kapsamı").
  const hpOptions = useMemo(
    () =>
      form.brand && form.model && form.engineHacmiBucket !== null
        ? getHpOptions(form.brand, form.model, form.engineHacmiBucket, form.yakitTuru)
        : [],
    [form.brand, form.model, form.engineHacmiBucket, form.yakitTuru]
  );

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
    setCarImageUrls([]);
    try {
      // Görsel araması opsiyoneldir ve kendi hatalarını yutar (bkz.
      // requestCarImageUrls) — fiyat tahminini asla bloklamamalı/bozmamalı.
      const [res, imageUrls] = await Promise.all([
        requestPrediction(input),
        requestCarImageUrls(input),
      ]);
      setResult(res);
      setCarImageUrls(imageUrls);
      setStatus("success");
      requestAnimationFrame(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Beklenmeyen bir hata oluştu.");
      setStatus("error");
    }
  }

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
        <div className="flex flex-col gap-8">
          {/* Kullanıcı bir aracı düşünürken doğal olarak izlediği sıra:
              önce ne olduğu (kimlik), sonra nasıl çalıştığı (motor),
              nasıl göründüğü (özellikler), ne kadar kullanıldığı, en
              sonda da durumu/hasarı. */}
          <FormSection title="Araç Kimliği">
            <VehicleSelector
              value={{
                brand: form.brand,
                model: form.model,
                engineHacmiBucket: form.engineHacmiBucket,
                engineExactCc: form.engineExactCc,
                yakitTuru: form.yakitTuru,
                trim: form.trim,
              }}
              onChange={setVehicle}
              brandError={errors.brand}
              modelError={errors.model}
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
          </FormSection>

          <FormSection title="Motor Bilgileri" divider>
            {form.yakitTuru ? (
              <LockedField
                label="Yakıt Türü"
                value={form.fuelType}
                hint="Seçilen motordan otomatik dolduruldu"
              />
            ) : (
              <SelectField
                id="fuelType"
                label="Yakıt Türü"
                value={form.fuelType}
                onChange={(v) => set("fuelType", v)}
                options={FUEL_TYPES}
                error={errors.fuelType}
                placeholder="Yakıt türü seçin"
              />
            )}
            <SelectField
              id="transmission"
              label="Vites"
              value={form.transmission}
              onChange={(v) => set("transmission", v)}
              options={TRANSMISSIONS}
              error={errors.transmission}
              placeholder="Vites seçin"
            />
            {form.engineExactCc !== null ? (
              <LockedField
                label="Motor Hacmi"
                value={`${form.engineExactCc} cc`}
                hint="Seçilen motordan otomatik dolduruldu"
              />
            ) : (
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
            )}
            {hpOptions.length === 1 ? (
              <LockedField
                label="Motor Gücü"
                value={`${hpOptions[0]} HP`}
                hint="Seçilen motordan otomatik dolduruldu"
              />
            ) : hpOptions.length > 1 ? (
              <SelectField
                id="enginePower"
                label="Motor Gücü"
                value={form.enginePower}
                onChange={(v) => set("enginePower", v)}
                options={hpOptions.map((hp) => `${hp}`)}
                error={errors.enginePower}
                placeholder={`Bu kombinasyon için ${hpOptions.length} geçerli değer var`}
              />
            ) : (
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
            )}
          </FormSection>

          <FormSection title="Araç Özellikleri" divider>
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
          </FormSection>

          <FormSection title="Kullanım Bilgisi" columns={1} divider>
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
          </FormSection>

          <FormSection title="Hasar Bilgisi" divider>
            <div className="sm:col-span-2">
              <YesNoField
                id="heavyDamage"
                label="Ağır Hasarlı"
                value={form.heavyDamage}
                onChange={(v) => set("heavyDamage", v)}
              />
            </div>
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
          </FormSection>
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
          <PredictionResult
            price={result.price}
            source={result.source}
            imageUrls={carImageUrls}
            carLabel={`${form.brand} ${form.model}`.trim()}
          />
        ) : null}
      </div>
    </div>
  );
}
