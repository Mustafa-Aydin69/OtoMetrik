import type { Metadata } from "next";
import { Archivo, Barlow_Condensed } from "next/font/google";
import "./globals.css";

// latin-ext, Türkçe glifler (ğ ş ı İ) için zorunlu — yalnızca "latin" alt
// kümesiyle bu karakterler sistem fontuna düşüyordu.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin", "latin-ext"],
});

// Barlow Condensed değişken font değil; ağırlıklar açıkça istenmeli.
const barlowCondensed = Barlow_Condensed({
  variable: "--font-barlow-condensed",
  subsets: ["latin", "latin-ext"],
  weight: ["600", "700"],
});

export const metadata: Metadata = {
  title: "OtoMetric AI — Aracının Gerçek Değerini Öğren",
  description:
    "Yapay zeka destekli ikinci el araç fiyat tahmini. Araç bilgilerini gir, gerçek piyasa değerini verilerle öğren.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="tr"
      className={`${archivo.variable} ${barlowCondensed.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#08090b] text-zinc-100">
        {children}
      </body>
    </html>
  );
}
