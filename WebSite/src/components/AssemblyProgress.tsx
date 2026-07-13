"use client";

/**
 * Montaj bölümünün alt kenarındaki ilerleme çubuğu + "kaydır" ipucu.
 * Değerler her frame'de ScrollAssemblySection'ın rAF döngüsünden ref'ler
 * üzerinden yazılır (React state kullanılmaz — performans gereksinimi).
 */
import type { RefObject } from "react";

export function AssemblyProgress({
  barRef,
  hintRef,
}: {
  barRef: RefObject<HTMLDivElement | null>;
  hintRef: RefObject<HTMLDivElement | null>;
}) {
  return (
    <>
      <div
        ref={hintRef}
        className="pointer-events-none absolute inset-x-0 bottom-10 flex flex-col items-center gap-2 text-zinc-300 transition-opacity duration-500"
      >
        <span className="text-xs font-medium uppercase tracking-[0.25em]">
          Kaydırarak montajı başlat
        </span>
        <svg
          aria-hidden
          className="size-5 animate-bounce"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </div>

      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 bg-white/10"
      >
        <div
          ref={barRef}
          className="h-full origin-left bg-sky-400/80"
          style={{ transform: "scaleX(0)" }}
        />
      </div>
    </>
  );
}
