/**
 * Tahmin formundaki mantıksal alan gruplarını (Araç Kimliği, Motor Bilgileri,
 * vb.) küçük bir başlık + responsive grid ile sarmalayan reusable bileşen.
 * `columns=1` tek alanlı bölümler (örn. Kilometre) için grid'i tek sütuna
 * sabitler — aksi halde tek çocuk 2 sütunlu grid'de yarım kalır.
 */
import type { ReactNode } from "react";

export function FormSection({
  title,
  columns = 2,
  divider = false,
  children,
}: {
  title: string;
  columns?: 1 | 2;
  /** Önceki bölümden görsel olarak ayırmak için üstte ince bir çizgi. */
  divider?: boolean;
  children: ReactNode;
}) {
  return (
    <section className={divider ? "border-t border-white/5 pt-8" : undefined}>
      <h3 className="mb-5 text-xs font-semibold uppercase tracking-[0.15em] text-zinc-500">
        {title}
      </h3>
      <div
        className={`grid grid-cols-1 gap-5 ${columns === 2 ? "sm:grid-cols-2" : ""}`}
      >
        {children}
      </div>
    </section>
  );
}
