"use client";

/**
 * PredictionForm'un kullandığı küçük, erişilebilir form alanı bileşenleri.
 * Ortak stil + label + hata mesajı kalıbını tek yerde tutar.
 */
import type { ReactNode } from "react";

const inputBase =
  "w-full rounded-lg border border-white/10 bg-white/[0.04] px-3.5 py-2.5 text-sm " +
  "text-zinc-100 placeholder:text-zinc-500 outline-none transition-colors " +
  "focus:border-sky-400/60 focus:bg-white/[0.06] focus:ring-2 focus:ring-sky-400/20";

function FieldShell({
  id,
  label,
  error,
  children,
}: {
  id: string;
  label: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={id}
        className="text-xs font-medium uppercase tracking-wider text-zinc-400"
      >
        {label}
      </label>
      {children}
      {error ? (
        <p id={`${id}-error`} role="alert" className="text-xs text-red-400">
          {error}
        </p>
      ) : null}
    </div>
  );
}

interface BaseProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  placeholder?: string;
}

export function TextField({
  id,
  label,
  value,
  onChange,
  error,
  placeholder,
  suggestions,
}: BaseProps & { suggestions?: readonly string[] }) {
  const listId = suggestions && suggestions.length > 0 ? `${id}-list` : undefined;
  return (
    <FieldShell id={id} label={label} error={error}>
      <input
        id={id}
        type="text"
        className={inputBase}
        value={value}
        placeholder={placeholder}
        list={listId}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-error` : undefined}
        onChange={(e) => onChange(e.target.value)}
      />
      {listId ? (
        <datalist id={listId}>
          {suggestions!.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      ) : null}
    </FieldShell>
  );
}

export function NumberField({
  id,
  label,
  value,
  onChange,
  error,
  placeholder,
  min,
  max,
  step,
  suffix,
}: BaseProps & { min?: number; max?: number; step?: number; suffix?: string }) {
  return (
    <FieldShell id={id} label={label} error={error}>
      <div className="relative">
        <input
          id={id}
          type="number"
          inputMode="numeric"
          className={`${inputBase} ${suffix ? "pr-14" : ""}`}
          value={value}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? `${id}-error` : undefined}
          onChange={(e) => onChange(e.target.value)}
        />
        {suffix ? (
          <span className="pointer-events-none absolute inset-y-0 right-3.5 flex items-center text-xs text-zinc-500">
            {suffix}
          </span>
        ) : null}
      </div>
    </FieldShell>
  );
}

export function SelectField({
  id,
  label,
  value,
  onChange,
  error,
  placeholder,
  options,
}: BaseProps & { options: readonly (string | number)[] }) {
  return (
    <FieldShell id={id} label={label} error={error}>
      <select
        id={id}
        className={`${inputBase} appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23a1a1aa%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22m6%209%206%206%206-6%22%2F%3E%3C%2Fsvg%3E')] bg-[position:right_0.875rem_center] bg-no-repeat pr-9`}
        value={value}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-error` : undefined}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="" disabled className="bg-zinc-900">
          {placeholder ?? "Seçin"}
        </option>
        {options.map((opt) => (
          <option key={opt} value={String(opt)} className="bg-zinc-900">
            {opt}
          </option>
        ))}
      </select>
    </FieldShell>
  );
}

export function YesNoField({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className="text-xs font-medium uppercase tracking-wider text-zinc-400">
        {label}
      </legend>
      <div
        className="grid grid-cols-2 gap-1 rounded-lg border border-white/10 bg-white/[0.04] p-1"
        role="radiogroup"
        aria-label={label}
      >
        {(
          [
            { text: "Hayır", val: false },
            { text: "Evet", val: true },
          ] as const
        ).map(({ text, val }) => (
          <label
            key={text}
            className={`flex cursor-pointer items-center justify-center rounded-md px-3 py-2 text-sm transition-colors ${
              value === val
                ? "bg-sky-500/20 text-sky-200"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <input
              type="radio"
              name={id}
              className="sr-only"
              checked={value === val}
              onChange={() => onChange(val)}
            />
            {text}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
