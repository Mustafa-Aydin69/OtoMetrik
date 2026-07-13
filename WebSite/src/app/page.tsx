import { ScrollAssemblySection } from "@/components/ScrollAssemblySection";
import { PredictionForm } from "@/components/PredictionForm";

const STEPS = [
  {
    title: "Bilgileri Gir",
    text: "Aracının marka, model, yıl, kilometre ve donanım bilgilerini forma gir.",
  },
  {
    title: "Model Analiz Etsin",
    text: "Binlerce gerçek ilandan öğrenen yapay zeka modeli aracını değerlendirsin.",
  },
  {
    title: "Değerini Gör",
    text: "Aracının güncel piyasa koşullarındaki tahmini değerini anında öğren.",
  },
] as const;

export default function Home() {
  return (
    <>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-white/[0.06] bg-[#08090b]/70 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
          <a href="#" className="text-sm font-semibold tracking-tight text-white">
            AutoValue <span className="text-sky-400">AI</span>
          </a>
          <nav className="flex items-center gap-6 text-sm text-zinc-400">
            <a href="#nasil-calisir" className="transition-colors hover:text-white">
              Nasıl Çalışır?
            </a>
            <a
              href="#form"
              className="rounded-lg bg-white/10 px-3.5 py-1.5 font-medium text-white transition-colors hover:bg-white/15"
            >
              Değerini Öğren
            </a>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <ScrollAssemblySection />

        <section
          id="nasil-calisir"
          className="relative mx-auto max-w-6xl scroll-mt-20 px-6 py-24"
        >
          <h2 className="text-center text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Nasıl Çalışır?
          </h2>
          <div className="mt-14 grid grid-cols-1 gap-6 sm:grid-cols-3">
            {STEPS.map((step, i) => (
              <div
                key={step.title}
                className="rounded-2xl border border-white/[0.07] bg-white/[0.03] p-6"
              >
                <span className="text-xs font-semibold tracking-widest text-sky-400/90">
                  0{i + 1}
                </span>
                <h3 className="mt-3 text-lg font-medium text-white">{step.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                  {step.text}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section id="form" className="relative scroll-mt-20 px-6 pb-28 pt-4">
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"
          />
          <div className="mx-auto max-w-6xl">
            <h2 className="text-center text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Aracının Değerini Öğren
            </h2>
            <p className="mx-auto mt-4 max-w-lg text-center text-sm leading-relaxed text-zinc-400">
              Bilgileri eksiksiz doldur; model, aracının güncel piyasa
              koşullarındaki tahmini değerini hesaplasın.
            </p>
            <div className="mt-12">
              <PredictionForm />
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/[0.06] px-6 py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 text-xs text-zinc-500 sm:flex-row">
          <span>© {new Date().getFullYear()} AutoValue AI</span>
          <span>Tahminler istatistikseldir; kesin satış fiyatı değildir.</span>
        </div>
      </footer>
    </>
  );
}
