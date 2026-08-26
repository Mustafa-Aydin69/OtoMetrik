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
  getBodyTypesForModel,
  getHpOptions,
  getVitesOptionsForModel,
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

type MacroStep = "vehicle" | "details";

export function PredictionForm() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [status, setStatus] = useState<Status>("idle");
  const [apiError, setApiError] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [carImageUrls, setCarImageUrls] = useState<string[]>([]);
  const resultRef = useRef<HTMLDivElement>(null);
  // "vehicle": VehicleSelector'ın tam sayfa Kategori›Marka›Model›Motor›Paket
  // akışı - "details": geri kalan tüm alanlar (Yıl, Yakıt, Vites, Kasa, Renk,
  // Km, Hasar) tek ekranda. VehicleSelector paket seçilince/atlanınca
  // onComplete() ile buraya geçer (bkz. setVehicle altındaki VehicleSelector
  // render'ı). resetKey, "Formu Temizle" sonrası VehicleSelector'ı SIFIRDAN
  // mount ederek kendi iç state'ini (kategori/kademe/arama) de temizler -
  // FormState'in aksine bunlar VehicleSelector'ın local state'idir.
  const [macroStep, setMacroStep] = useState<MacroStep>("vehicle");
  const [resetKey, setResetKey] = useState(0);

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
    // Kasa tipi marka+model'e bağlı (motor/paket'ten bağımsız) - sadece marka
    // veya model gerçekten değiştiyse sıfırlanır, aksi halde kullanıcının
    // motor/paket adımlarında yaptığı seçim kasa tipini bozmaz.
    const vehicleChanged = next.brand !== form.brand || next.model !== form.model;
    const bodyOptions = next.brand && next.model ? getBodyTypesForModel(next.brand, next.model) : [];
    const vitesOptions = next.brand && next.model ? getVitesOptionsForModel(next.brand, next.model) : [];

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
      bodyType: vehicleChanged ? (bodyOptions.length === 1 ? bodyOptions[0] : "") : prev.bodyType,
      transmission: vehicleChanged ? (vitesOptions.length === 1 ? vitesOptions[0] : "") : prev.transmission,
    }));
    setErrors((prev) => {
      const rest = { ...prev };
      delete rest.brand;
      delete rest.model;
      delete rest.fuelType;
      delete rest.engineDisplacement;
      delete rest.enginePower;
      if (vehicleChanged) {
        delete rest.bodyType;
        delete rest.transmission;
      }
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

  // Kasa Tipi'nin bu marka+model için kaç geçerli değeri var: 0 -> tam
  // kanonik listeye (BODY_TYPES) serbest seçim (araç henüz seçilmedi VEYA bu
  // marka+model için veri yok), 1 -> LockedField, >1 -> yalnızca gerçekten
  // görülen değerlerden oluşan SelectField (bkz. plan: HP ile aynı desen).
  const bodyTypeOptions = useMemo(
    () => (form.brand && form.model ? getBodyTypesForModel(form.brand, form.model) : []),
    [form.brand, form.model]
  );

  // Vites'in bu marka+model için kaç geçerli değeri var: BODY_TYPE ile aynı
  // desen (bkz. yukarısı) - 0 -> tam kanonik listeye (TRANSMISSIONS) serbest
  // seçim, 1 -> LockedField, >1 -> yalnızca gerçekten görülen değerlerden
  // oluşan SelectField.
  const vitesOptions = useMemo(
    () => (form.brand && form.model ? getVitesOptionsForModel(form.brand, form.model) : []),
    [form.brand, form.model]
  );

  function resetForm() {
    setForm(INITIAL);
    setErrors({});
    setStatus("idle");
    setApiError("");
    setResult(null);
    setCarImageUrls([]);
    setMacroStep("vehicle");
    setResetKey((k) => k + 1);
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
        <div className="mb-6 flex justify-end">
          <button
            type="button"
            onClick={resetForm}
            className="text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-200"
          >
            Formu Temizle
          </button>
        </div>

        {macroStep === "vehicle" ? (
          <VehicleSelector
            key={resetKey}
            value={{
              brand: form.brand,
              model: form.model,
              engineHacmiBucket: form.engineHacmiBucket,
              engineExactCc: form.engineExactCc,
              yakitTuru: form.yakitTuru,
              trim: form.trim,
            }}
            onChange={setVehicle}
            onComplete={() => setMacroStep("details")}
            brandError={errors.brand}
            modelError={errors.model}
          />
        ) : (
        <div className="flex flex-col gap-8">
          {/* Araç zaten seçildi (önceki makro-adım) - burada özet + değiştirme
              linki, ardından kullanıcı bir aracı düşünürken doğal olarak
              izlediği sıra: nasıl çalıştığı (motor), nasıl göründüğü
              (özellikler), ne kadar kullanıldığı, en sonda da durumu/hasarı. */}
          <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
            <span className="truncate text-sm text-zinc-200">
              {[form.brand, form.model, form.trim].filter(Boolean).join(" · ")}
            </span>
            <button
              type="button"
              onClick={() => setMacroStep("vehicle")}
              className="shrink-0 text-xs font-medium text-sky-300 transition-colors hover:text-sky-200"
            >
              Aracı Değiştir
            </button>
          </div>

          <FormSection title="Araç Bilgileri" columns={1}>
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
            {vitesOptions.length === 1 ? (
              <LockedField
                label="Vites"
                value={vitesOptions[0]}
                hint="Seçilen araçtan otomatik dolduruldu"
              />
            ) : (
              <SelectField
                id="transmission"
                label="Vites"
                value={form.transmission}
                onChange={(v) => set("transmission", v)}
                options={vitesOptions.length > 0 ? vitesOptions : TRANSMISSIONS}
                error={errors.transmission}
                placeholder={
                  vitesOptions.length > 0
                    ? `Bu araç için ${vitesOptions.length} geçerli vites var`
                    : "Vites seçin"
                }
              />
            )}
            {form.fuelType === "Elektrik" ? null : form.engineExactCc !== null ? (
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
                placeholder="örn. 1600"
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
            {bodyTypeOptions.length === 1 ? (
              <LockedField
                label="Kasa Tipi"
                value={bodyTypeOptions[0]}
                hint="Seçilen araçtan otomatik dolduruldu"
              />
            ) : (
              <SelectField
                id="bodyType"
                label="Kasa Tipi"
                value={form.bodyType}
                onChange={(v) => set("bodyType", v)}
                options={bodyTypeOptions.length > 0 ? bodyTypeOptions : BODY_TYPES}
                error={errors.bodyType}
                placeholder={
                  bodyTypeOptions.length > 0
                    ? `Bu araç için ${bodyTypeOptions.length} geçerli kasa tipi var`
                    : "Kasa tipi seçin"
                }
              />
            )}
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

          {status === "error" ? (
            <div
              role="alert"
              className="flex flex-col items-start gap-3 rounded-lg border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-300 sm:flex-row sm:items-center sm:justify-between"
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
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-50 px-6 py-3.5 text-sm font-semibold text-zinc-950 transition-all hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
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
        </div>
        )}
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
