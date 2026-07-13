"use client";

/**
 * Sayfanın ana deneyimi: 600vh'lik scroll alanı boyunca sticky kalan video,
 * scroll ilerlemesi → video.currentTime eşlemesi (rAF ile yumuşatılmış,
 * iki yönde simetrik), scroll'a bağlı başlıklar ve form geçişi.
 *
 * Performans kuralları:
 * - Scroll sırasında React state GÜNCELLENMEZ; her şey ref + style ile yazılır.
 * - video.currentTime yalnızca rAF döngüsünde ve seek meşgul değilken yazılır.
 * - ScrollTrigger yalnızca progress kaynağı olarak kullanılır (pin CSS sticky).
 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ScrollScrubVideo } from "./ScrollScrubVideo";
import { AssemblyProgress } from "./AssemblyProgress";

const CAPTIONS = [
  { text: "Bir otomobil yalnızca parçalardan oluşmaz.", from: 0.03, to: 0.22 },
  { text: "Geçmişi, donanımı ve durumu değerini belirler.", from: 0.42, to: 0.62 },
  { text: "Gerçek değerini verilerle öğren.", from: 0.8, to: 0.97 },
] as const;

/** from–to aralığında kenarlarda yumuşayan (trapez) opaklık eğrisi. */
function captionOpacity(p: number, from: number, to: number): number {
  if (p <= from || p >= to) return 0;
  const edge = (to - from) * 0.3;
  return Math.min(1, (p - from) / edge, (to - p) / edge);
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function subscribeReducedMotion(onChange: () => void) {
  const mql = window.matchMedia(REDUCED_MOTION_QUERY);
  mql.addEventListener("change", onChange);
  return () => mql.removeEventListener("change", onChange);
}

/** SSR'da false; client'ta canlı media-query değeri. */
function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeReducedMotion,
    () => window.matchMedia(REDUCED_MOTION_QUERY).matches,
    () => false
  );
}

type VideoState = "loading" | "ready" | "error";

export function ScrollAssemblySection() {
  const wrapRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const hintRef = useRef<HTMLDivElement>(null);
  const captionRefs = useRef<(HTMLParagraphElement | null)[]>([]);

  const targetProgress = useRef(0);
  const smoothProgress = useRef(0);

  const [videoState, setVideoState] = useState<VideoState>("loading");
  const reducedMotion = useReducedMotion();

  const onVideoReady = useCallback(() => setVideoState("ready"), []);
  const onVideoError = useCallback(() => setVideoState("error"), []);

  const scrubActive = !reducedMotion && videoState !== "error";

  useEffect(() => {
    if (!scrubActive || !wrapRef.current) return;

    gsap.registerPlugin(ScrollTrigger);

    const st = ScrollTrigger.create({
      trigger: wrapRef.current,
      start: "top top",
      end: "bottom bottom",
      onUpdate: (self) => {
        targetProgress.current = self.progress;
      },
      // Sayfa yenilendiğinde / resize'da mevcut scroll konumuyla anında
      // senkronize ol (yumuşatma atlanır ki doğru frame'e oturulsun).
      onRefresh: (self) => {
        targetProgress.current = self.progress;
        smoothProgress.current = self.progress;
      },
    });

    let rafId = 0;
    const tick = () => {
      const target = targetProgress.current;
      let smooth = smoothProgress.current;
      smooth += (target - smooth) * 0.14;
      if (Math.abs(target - smooth) < 0.0004) smooth = target;
      smoothProgress.current = smooth;

      // Video frame'ini scroll konumuna sabitle (iki yönde simetrik).
      const video = videoRef.current;
      if (
        video &&
        video.readyState >= 1 &&
        Number.isFinite(video.duration) &&
        video.duration > 0 &&
        !video.seeking
      ) {
        const time = smooth * Math.max(video.duration - 0.05, 0);
        if (Math.abs(video.currentTime - time) > 0.001) {
          video.currentTime = time;
        }
      }

      // Başlıklar.
      captionRefs.current.forEach((el, i) => {
        if (!el) return;
        const { from, to } = CAPTIONS[i];
        const o = captionOpacity(smooth, from, to);
        el.style.opacity = o.toFixed(3);
        el.style.transform = `translateY(${(1 - o) * 14}px)`;
      });

      // İlerleme çubuğu + ipucu.
      if (barRef.current) barRef.current.style.transform = `scaleX(${smooth})`;
      if (hintRef.current) hintRef.current.style.opacity = smooth > 0.02 ? "0" : "1";

      // Sona yaklaşırken sinematik çözülme: hafif küçülme + karartma + blur —
      // form bölümü bunun üzerinden yükselir.
      if (stageRef.current) {
        const k = clamp01((smooth - 0.86) / 0.14);
        stageRef.current.style.transform = `scale(${1 - 0.07 * k})`;
        stageRef.current.style.filter = `brightness(${1 - 0.5 * k}) blur(${(k * 5).toFixed(1)}px)`;
      }

      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      st.kill();
    };
  }, [scrubActive]);

  // --- prefers-reduced-motion veya video yok: statik sade sunum ---
  if (reducedMotion || videoState === "error") {
    return (
      <section
        aria-label="Tanıtım"
        className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center"
      >
        <div
          aria-hidden
          className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_40%,rgba(56,130,190,0.12),transparent)]"
        />
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-6xl">
          Bir otomobil yalnızca parçalardan oluşmaz.
        </h1>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-zinc-400">
          Geçmişi, donanımı ve durumu değerini belirler. Gerçek değerini
          verilerle öğren.
        </p>
        <a
          href="#form"
          className="mt-10 rounded-xl bg-zinc-50 px-6 py-3.5 text-sm font-semibold text-zinc-950 transition-colors hover:bg-white"
        >
          Değerini Öğren
        </a>
      </section>
    );
  }

  return (
    <section
      ref={wrapRef}
      aria-label="Araç montaj animasyonu"
      className="relative h-[600vh]"
    >
      <div className="sticky top-0 h-screen overflow-hidden">
        {/* Video sahnesi */}
        <div ref={stageRef} className="absolute inset-0 will-change-transform">
          <ScrollScrubVideo
            ref={videoRef}
            className="absolute inset-0 size-full object-cover"
            onReady={onVideoReady}
            onError={onVideoError}
          />
          {/* Kenar vinyeti — aracı kapatmadan premium stüdyo hissi */}
          <div
            aria-hidden
            className="absolute inset-0 bg-[radial-gradient(ellipse_75%_65%_at_50%_50%,transparent_55%,rgba(5,6,8,0.75))]"
          />
        </div>

        {/* Yükleme göstergesi */}
        {videoState === "loading" ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div
              className="size-8 animate-spin rounded-full border-2 border-white/15 border-t-white/70"
              role="status"
              aria-label="Video yükleniyor"
            />
          </div>
        ) : null}

        {/* Scroll'a bağlı başlıklar */}
        {CAPTIONS.map((c, i) => (
          <p
            key={c.text}
            ref={(el) => {
              captionRefs.current[i] = el;
            }}
            className={`pointer-events-none absolute inset-x-0 px-6 text-center text-2xl font-medium tracking-tight text-white/95 [text-shadow:0_2px_24px_rgba(0,0,0,0.7)] sm:text-4xl ${
              i === 0 ? "top-[16%]" : i === 1 ? "top-[14%]" : "top-[42%]"
            }`}
            style={{ opacity: 0 }}
          >
            {c.text}
          </p>
        ))}

        {videoState === "ready" ? (
          <AssemblyProgress barRef={barRef} hintRef={hintRef} />
        ) : null}
      </div>
    </section>
  );
}
