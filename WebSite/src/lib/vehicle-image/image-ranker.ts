/**
 * Aday görselleri sorgu ile ne kadar örtüştüğüne göre sıralar.
 *
 * Gerçek bir CLIP (image-text similarity) modeli burada YOK: yerel
 * çalıştırma yüz(lerce) MB model indirmesi + WASM/ONNX çıkarım gerektirir,
 * bu da bir Next.js API route'unda (özellikle serverless barındırmada)
 * süre/bellek sınırlarını zorlar; harici bir CLIP API'si ise (Replicate,
 * HF Inference) yine hesap/anahtar gerektirir. Bunun yerine, Commons'ın
 * döndürdüğü başlık/açıklama/kategori metniyle sorgu kelimelerinin
 * örtüşümüne dayanan basit ama gerçek bir skorlama kullanılıyor — "ilk
 * sonucu göster"den daha isabetli, sıfır ek bağımlılık.
 *
 * ImageRanker arayüzü, ileride gerçek bir CLIP tabanlı sıralayıcının
 * (örn. ayrı bir inference servisine istek atan bir implementasyon)
 * bu arayüzü uygulayıp doğrudan yerine takılabilmesi için tasarlandı.
 */
import type { ImageCandidate, RankedImageCandidate } from "./types";

export interface ImageRanker {
  rank(candidates: ImageCandidate[], queryTokens: string[]): RankedImageCandidate[];
}

function scoreOverlap(sourceText: string, tokens: string[]): number {
  if (!sourceText || tokens.length === 0) return 0;
  let score = 0;
  for (const token of tokens) {
    if (token.length < 2) continue;
    if (sourceText.includes(token.toLowerCase())) score += 1;
  }
  return score;
}

export const lexicalRanker: ImageRanker = {
  rank(candidates, tokens) {
    return candidates
      .map((c) => ({ ...c, score: scoreOverlap(c.sourceText, tokens) }))
      .sort((a, b) => b.score - a.score);
  },
};
