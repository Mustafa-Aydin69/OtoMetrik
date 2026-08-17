/**
 * Confidence Gate - hard-reject'ten sağ çıkan ve skorlanan adaylardan hangisi
 * (hangileri) kullanıcıya gösterilecek kadar güvenilir, bunu karar verir
 * (bkz. plan Madde 15). Asıl karar mekanizması candidate-ranker.ts'teki
 * ağırlıklı skorlama + hard-reject'tir - buradaki zero-overlap kontrolü
 * yalnızca SON güvenlik ağıdır (bkz. plan Madde 14), tek başına yeterli
 * DEĞİLDİR.
 *
 * confidence = candidate.score / MAX_SCORE (0-1), generation'a özel bonus/
 * ceza ile ayarlanır (bkz. plan Madde 16 - "generation doğrulanmışsa
 * confidence ciddi artmalı, target generation biliniyor ama adayda hiç
 * doğrulanamadıysa ihtiyatlı düşülmeli").
 *
 * Eşikler (LOW/MEDIUM/HIGH) mevcut WEIGHTS'e göre kalibre edilmiş başlangıç
 * değerleridir - gerçek kullanımda ayarlanabilir (bkz. plan: "kesin
 * threshold'ları scoring sistemine göre kalibre et").
 */
import { MAX_SCORE, searchableText } from "./candidate-ranker";
import { identityTokens } from "./query-generator";
import type { ImageSelectionResult, RankedImageCandidate, VehicleIdentity } from "./types";

// "Zero-overlap" güvenlik ağında SAYILMAYACAK genel/stopword tokenlar (bkz.
// plan Madde 14) - marka/model/generation gibi anlamlı alanlar buraya
// GİRMEZ, yalnızca hiçbir araca özgü olmayan genel kelimeler.
const STOPWORDS = new Set([
  "car", "vehicle", "automobile", "image", "file", "photo", "photograph",
  "the", "a", "an", "of", "in", "and", "with", "on", "at",
]);

const LOW_THRESHOLD = 0.3;

function meaningfulTokensFor(vehicle: VehicleIdentity): string[] {
  return identityTokens(vehicle)
    .map((t) => t.toLowerCase())
    .filter((t) => t.length >= 2 && !STOPWORDS.has(t));
}

function hasMeaningfulOverlap(text: string, meaningfulTokens: string[]): boolean {
  if (meaningfulTokens.length === 0) return true; // hiçbir kimlik bilgisi yoksa kontrol anlamsız, engelleme
  return meaningfulTokens.some((t) => text.includes(t));
}

function computeConfidence(candidate: RankedImageCandidate, vehicle: VehicleIdentity): number {
  let raw = candidate.score / MAX_SCORE;
  if (vehicle.generation) {
    // generation BİLİNİYOR: adayda DOĞRULANDIYSA ekstra güven, DOĞRULANAMADIYSA
    // ihtiyatlı düş (candidate-ranker zaten AÇIK çelişkileri hard-reject etti -
    // burası yalnızca "susma/belirsizlik" durumunu cezalandırıyor).
    raw = candidate.matched.generation ? Math.min(1, raw * 1.15) : raw * 0.7;
  }
  return Math.max(0, Math.min(1, raw));
}

function toSelectionResult(candidate: RankedImageCandidate | null, confidence: number, rejectedCount: number): ImageSelectionResult {
  if (!candidate) return { imageUrl: null, confidence: 0, rejectedCount };
  return {
    imageUrl: confidence >= LOW_THRESHOLD ? candidate.url : null,
    confidence,
    candidateScore: candidate.score,
    matched: candidate.matched,
    rejectedCount,
    source: candidate.source,
  };
}

export interface ConfidentSelection {
  /** Güven eşiğini geçen, en iyiden en kötüye sıralı URL listesi - frontend'in cascading onError fallback'i için (bkz. PredictionResult.tsx). */
  urls: string[];
  /** En iyi adayın tam seçim sonucu - debug log/observability için (bkz. plan Madde 19-20). */
  best: ImageSelectionResult;
}

/**
 * rankCandidates() çıktısından, güven eşiğini geçen adayları (en iyiden en
 * kötüye) seçer. Reddedilenler (rejected:true) ve zero-meaningful-overlap
 * güvenlik ağını geçemeyenler HİÇ döndürülmez - "hiç fotoğraf göstermemek,
 * yanlış fotoğraf göstermekten iyidir" (bkz. plan Madde 15/38).
 */
export function selectConfidentImages(ranked: RankedImageCandidate[], vehicle: VehicleIdentity): ConfidentSelection {
  const accepted = ranked.filter((c) => !c.rejected);
  const rejectedCount = ranked.length - accepted.length;
  const meaningfulTokens = meaningfulTokensFor(vehicle);

  if (accepted.length === 0) {
    return { urls: [], best: toSelectionResult(null, 0, rejectedCount) };
  }

  const confident = accepted
    .map((c) => ({ candidate: c, confidence: computeConfidence(c, vehicle), text: searchableText(c) }))
    .filter(({ confidence, text }) => confidence >= LOW_THRESHOLD && hasMeaningfulOverlap(text, meaningfulTokens));

  const best = toSelectionResult(confident[0]?.candidate ?? null, confident[0]?.confidence ?? 0, rejectedCount);
  return { urls: confident.map((c) => c.candidate.url), best };
}
