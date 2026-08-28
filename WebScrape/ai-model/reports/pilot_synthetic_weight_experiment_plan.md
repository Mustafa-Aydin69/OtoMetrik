# Sentetik sample_weight ablation deney planı (Faz 30 pilot) — TASARIM, HENÜZ ÇALIŞTIRILMADI

## Ön koşul
LightGBM'in scikit-learn API'si (`LGBMRegressor.fit`) `sample_weight` parametresini
native destekler — `train.py`'nin `train_final_model()`'ında ek kod gerektirmez,
sadece `model.fit(X, y, sample_weight=w)` çağrısına `w` eklenir.

## Deney tasarımı
1. **Split önce, sentetik sonra.** `preprocess.split_features_target()` (veya
   `q99_and_hp_retrain_comparison.py`'deki 80/20 deneyiyle aynı `random_state=42`)
   ÖNCE gerçek veri üzerinde çalıştırılır. Sentetik satırlar SADECE train
   partisyonuna eklenir — test/holdout kesinlikle sentetik görmez (kullanıcı
   kuralı — bu, split'i sentetik satırlar eklendikten SONRA değil ÖNCE yapmayı
   *zorunlu* kılar, aksi halde sentetik bir satır türediği gerçek satırla aynı
   partisyona düşmeyebilir ve sızıntı riski oluşur).
2. **hierarchical_price OOF'a sentetik dahil edilmez.** `compute_oof_feature()`
   sadece GERÇEK satırlarla çalışmalı — sentetik satırların `brand_model_median_price`
   değeri, gerçek OOF lookup'tan (`attach_lookup_feature`) alınır, kendi
   fold'larına dahil edilip kendi fiyatını sızdırmaz.
3. **3 paralel model, tek fark `sample_weight`:**
   - Baseline: sadece gerçek veri (sentetik yok) — referans.
   - W=0.25: gerçek satırlar weight=1.0, pilot sentetik satırlar weight=0.25.
   - W=0.50: aynı, sentetik weight=0.50.
   - W=1.00: aynı, sentetik weight=1.00 (gerçek ile eşit ağırlık).
4. **Değerlendirme:** `q99_and_hp_retrain_comparison.py`'deki segment tablosu
   (overall, >5M TL, premium/luxury, support bucket) + AYRICA münhasıran
   Ferrari 458 / Huracan / Ghost için MAE/MAPE — hem TEST setindeki gerçek
   satırlar üzerinden (varsa) hem de eğitim-içi (in-sample) tahmin tutarlılığı.
5. **Kabul kriteri (öneri, kullanıcı onayına açık):** W arttıkça premium/rare
   segment hatası düşerken overall/mainstream segment MAE'si anlamlı
   bozulmuyorsa (Faz 29'daki "overall'ı ezmesin" kriteriyle aynı mantık) en
   düşük yeterli W seçilir — sentetik veriye gerçek veriden DAHA FAZLA ağırlık
   vermek (W>1.0) bu pilotta denenmez, kullanıcının "sentetik veri gerçek
   veriyi ezmemeli" kuralına aykırı olur.

## Riskler / izlenecekler
- n=18 sentetik satır, 3 grup için — istatistiksel olarak KÜÇÜK bir pilot;
  sonuçlar yön gösterir ama kesin değildir, toplu üretim kararına tek başına
  yeterli kanıt SAYILMAMALI.
- `source_parent_ids` her satırda saklandığı için, W deneyinden sonra "hangi
  gerçek satır en çok etkiledi" geriye izlenebilir (yorumlanabilirlik).
- Bu belge sadece plandır — retrain/deney kullanıcı onayı olmadan ÇALIŞTIRILMAZ.
