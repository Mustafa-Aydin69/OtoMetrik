"use client";

/**
 * Scroll ile "scrub" edilen video elemanı. currentTime'ı BU bileşen değil,
 * üst bileşen (ScrollAssemblySection) rAF döngüsünde ref üzerinden yazar.
 * Bu bileşen yalnızca yükleme/hata yaşam döngüsünü raporlar.
 *
 * Asset yolu: /videos/assembly.webm + /videos/assembly.mp4
 * (bkz. public/videos/README.md — Higgsfield çıktısı buraya konur)
 */
import { forwardRef, useEffect, useRef } from "react";

export const VIDEO_SOURCES = {
  webm: "/videos/assembly.webm",
  mp4: "/videos/assembly.mp4",
  poster: "/videos/poster.jpg",
} as const;

interface Props {
  className?: string;
  onReady: () => void;
  onError: () => void;
}

export const ScrollScrubVideo = forwardRef<HTMLVideoElement, Props>(
  function ScrollScrubVideo({ className, onReady, onError }, ref) {
    const readyFired = useRef(false);
    const errorFired = useRef(false);
    const localRef = useRef<HTMLVideoElement | null>(null);

    useEffect(() => {
      const video = localRef.current;
      if (!video) return;

      const fireReady = () => {
        if (
          !readyFired.current &&
          Number.isFinite(video.duration) &&
          video.duration > 0
        ) {
          readyFired.current = true;
          onReady();
        }
      };
      const fireError = () => {
        if (!errorFired.current) {
          errorFired.current = true;
          onError();
        }
      };

      // Metadata zaten yüklendiyse (bfcache/geri dönüş) hemen bildir.
      if (video.readyState >= 1) fireReady();

      video.addEventListener("loadedmetadata", fireReady);
      video.addEventListener("error", fireError);

      // Kaynak dosyalar hiç yoksa 'error' son <source> elemanında oluşur ve
      // video elemanına bubble etmez; güvence olarak network durumunu kontrol
      // et (NETWORK_NO_SOURCE = 3).
      const probe = window.setTimeout(() => {
        if (video.readyState === 0 && video.networkState === 3) fireError();
      }, 4000);

      return () => {
        video.removeEventListener("loadedmetadata", fireReady);
        video.removeEventListener("error", fireError);
        window.clearTimeout(probe);
      };
    }, [onReady, onError]);

    return (
      <video
        ref={(el) => {
          localRef.current = el;
          if (typeof ref === "function") ref(el);
          else if (ref) ref.current = el;
        }}
        className={className}
        muted
        playsInline
        preload="auto"
        poster={VIDEO_SOURCES.poster}
        aria-label="1969 Ford Mustang Fastback montaj animasyonu"
        disablePictureInPicture
      >
        <source src={VIDEO_SOURCES.webm} type="video/webm" />
        <source src={VIDEO_SOURCES.mp4} type="video/mp4" />
      </video>
    );
  }
);
