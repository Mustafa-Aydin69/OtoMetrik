import { ScrollAssemblySection } from "@/components/ScrollAssemblySection";
import { PredictionForm } from "@/components/PredictionForm";

/**
 * Modelin kullandığı girdiler ve her birinin fiyatı hangi yönde çektiği.
 * Formdaki alanlarla birebir aynı sırada — kullanıcı aşağı indiğinde
 * doldurmaya başladığı alanların neden sorulduğunu bilmiş oluyor.
 */
const INPUTS = [
  {
    name: "Marka & model",
    effect: "Fiyatın temel bandını belirler.",
  },
  {
    name: "Model yılı",
    effect: "Amortisman eğrisinde nerede olduğunu söyler.",
  },
  {
    name: "Kilometre",
    effect: "Aynı yılda 60.000 km fark, bandın içinde ciddi oynama demek.",
  },
  {
    name: "Motor & vites",
    effect: "Aynı kasada her kombinasyon ayrı fiyatlanıyor.",
  },
  {
    name: "Donanım paketi",
    effect: "Paketler arası fark modele ayrı girdi olarak veriliyor.",
  },
  {
    name: "Hasar & boya kaydı",
    effect: "Kayıt varsa model bunu ayrı bir düşüş olarak öğrendi.",
  },
] as const;

export default function Home() {
  return (
    <>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-white/[0.06] bg-[#08090b]/70 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
          <a href="#" className="text-sm font-semibold tracking-tight text-white">
            OtoMetric <span className="text-accent">AI</span>
          </a>
          <nav className="flex items-center gap-6 text-sm text-zinc-400">
            <a href="#model" className="transition-colors hover:text-white">
              Model neye bakıyor?
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
          id="model"
          className="relative mx-auto max-w-6xl scroll-mt-20 px-6 py-24"
        >
          <h2 className="font-display text-4xl font-semibold text-white sm:text-5xl">
            Model neye bakıyor?
          </h2>
          <p className="mt-4 max-w-xl text-sm leading-relaxed text-zinc-400">
            Altı girdi, fiyatı birbirinden bağımsız yönlere çekiyor. Eğitim
            verisi açık bir referans veri setinden, test ve doğrulama verisi ise
            ilan platformundan canlı toplanıyor.
          </p>
          <dl className="mt-12 grid grid-cols-1 gap-x-12 sm:grid-cols-2">
            {INPUTS.map((input) => (
              <div
                key={input.name}
                className="border-t border-white/[0.07] py-4"
              >
                <dt className="text-base font-medium text-white">
                  {input.name}
                </dt>
                <dd className="mt-1 text-sm leading-relaxed text-zinc-400">
                  {input.effect}
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <section id="form" className="relative scroll-mt-20 px-6 pb-28 pt-4">
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"
          />
          <div className="mx-auto max-w-6xl">
            <h2 className="text-center font-display text-4xl font-semibold text-white sm:text-5xl">
              Aracını tanımla
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
          <span>© {new Date().getFullYear()} OtoMetric AI</span>
          <span>Tahminler istatistikseldir; kesin satış fiyatı değildir.</span>
        </div>
      </footer>
    </>
  );
}
