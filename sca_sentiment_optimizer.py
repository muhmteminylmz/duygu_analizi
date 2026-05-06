"""
sca_sentiment_optimizer.py
==========================
Sine-Cosine Algorithm (SCA) tabanlı Duygu Analizi Ağırlık Optimizörü

Problem Tanımı:
    Twitter verileri üzerindeki melez duygu analizi modelinde, her tweet'ten
    önceden çıkarılmış 10 dilbilimsel/yapısal özelliğin duygu skoruna katkısını
    (ağırlığını) SCA ile optimize eder.

10 Kıstas (Girdi Özellikleri):
    1. Değilleme (Negation)         – olumsuzluk eki / "değil/yok" kelime oranı
    2. Yoğunlaştırıcılar (Intensifiers) – "çok", "aşırı" gibi şiddet kelimesi katsayısı
    3. Büyük Harf Oranı             – tamamı büyük harfli kelime oranı
    4. Noktalama Yoğunluğu          – ünlem/soru işareti oranı
    5. Kelime Çeşitliliği (TTR)     – benzersiz kelime / toplam kelime
    6. Fiil Zamanı (Tense)          – geçmiş/gelecek zaman eki oranı
    7. Öznellik Skoru (Subjectivity)– duygu belirten kelime yoğunluğu
    8. Sıfat ve Zarf Oranı          – sıfat/zarfların toplam kelimelere oranı
    9. İroni Tespiti (Irony)        – zıt kutuplu kelime eş-bulunma durumu
   10. Zıtlık Bağlaçları (Contrast) – "ama/lakin/fakat" gibi bağlaçların varlığı

SCA Güncelleme Kuralları:
    r1 = a - t * (a / t_max)          (a = 2)
    r2 ~ Uniform[0, 2π]
    r3 ~ Uniform[0, 2]
    r4 ~ Uniform[0, 1]

    if r4 < 0.5:
        w_{t+1} = w_t + r1 * sin(r2) * |r3 * P_best - w_t|
    else:
        w_{t+1} = w_t + r1 * cos(r2) * |r3 * P_best - w_t|

Uygunluk (Fitness) Fonksiyonu:
    MSE = (1/N) * Σ (y_i - ŷ_i)²   (minimize edilir)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ortalama Karesel Hata (Mean Squared Error)."""
    return float(np.mean((y_true - y_pred) ** 2))


def _linear_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Ağırlıklı doğrusal skor:  ŷ = X @ w  (matris çarpımı)."""
    return X @ w


# ---------------------------------------------------------------------------
# Ana Sınıf
# ---------------------------------------------------------------------------

class SCA_SentimentOptimizer:
    """
    Sine-Cosine Algorithm (SCA) ile Duygu Analizi Ağırlık Optimizörü.

    Parametreler
    ------------
    a : float
        SCA lineer azalma sabiti (varsayılan: 2).
    w_min : float
        Ağırlık vektörü alt sınırı (varsayılan: -5.0).
    w_max : float
        Ağırlık vektörü üst sınırı (varsayılan:  5.0).
    random_state : int veya None
        Tekrar edilebilirlik için tohum değeri.
    verbose : bool
        Her iterasyonda ilerleme bilgisi yazdır.

    Öznitelikler
    ------------
    best_weights_ : np.ndarray, şekil (n_features,)
        Optimizasyon sonrası en iyi ağırlık vektörü.
    best_fitness_ : float
        En iyi ajan için ulaşılan minimum MSE değeri.
    fitness_history_ : list[float]
        Her iterasyondaki en iyi uygunluk değerlerinin geçmişi.
    """

    # Özellik isimlerini sabit olarak tanımla (belgeleme amaçlı)
    FEATURE_NAMES = [
        "Değilleme (Negation)",
        "Yoğunlaştırıcılar (Intensifiers)",
        "Büyük Harf Oranı",
        "Noktalama Yoğunluğu",
        "Kelime Çeşitliliği (TTR)",
        "Fiil Zamanı (Tense)",
        "Öznellik Skoru (Subjectivity)",
        "Sıfat ve Zarf Oranı",
        "İroni Tespiti",
        "Zıtlık Bağlaçları",
    ]

    def __init__(
        self,
        a: float = 2.0,
        w_min: float = -5.0,
        w_max: float = 5.0,
        random_state: int | None = 42,
        verbose: bool = True,
    ) -> None:
        self.a = a
        self.w_min = w_min
        self.w_max = w_max
        self.random_state = random_state
        self.verbose = verbose

        # Eğitim sonrası doldurulacak öznitelikler
        self.best_weights_: np.ndarray | None = None
        self.best_fitness_: float = float("inf")
        self.fitness_history_: list[float] = []

    # ------------------------------------------------------------------
    # Eğitim (Optimizasyon)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        pop_size: int = 30,
    ) -> "SCA_SentimentOptimizer":
        """
        SCA ile ağırlık vektörünü optimize et.

        Parametreler
        ------------
        X : np.ndarray, şekil (n_samples, 10)
            10 dilbilimsel özelliği içeren özellik matrisi.
        y : np.ndarray, şekil (n_samples,)
            Etiketler: sürekli skor veya {-1, 1} sınıf etiketleri.
        epochs : int
            Maksimum iterasyon sayısı (t_max).
        pop_size : int
            Arama ajanı (popülasyon) sayısı.

        Döndürür
        --------
        self : SCA_SentimentOptimizer
            Eğitilmiş nesne (akıcı zincirleme için).
        """
        rng = np.random.default_rng(self.random_state)

        n_features = X.shape[1]
        t_max = epochs

        # ---------------------------------------------------------------
        # 1. Popülasyonu başlat: her ajan 10 boyutlu bir ağırlık vektörü
        # ---------------------------------------------------------------
        # population şekli: (pop_size, n_features)
        population = rng.uniform(self.w_min, self.w_max, size=(pop_size, n_features))

        # ---------------------------------------------------------------
        # 2. İlk uygunluk değerlerini hesapla
        # ---------------------------------------------------------------
        fitness = self._evaluate_population(population, X, y)

        # En iyi ajan (P_best)
        best_idx = int(np.argmin(fitness))
        p_best = population[best_idx].copy()
        self.best_fitness_ = float(fitness[best_idx])
        self.fitness_history_ = [self.best_fitness_]

        # ---------------------------------------------------------------
        # 3. Ana optimizasyon döngüsü
        # ---------------------------------------------------------------
        for t in range(1, t_max + 1):
            # r1: lineer azalan kontrol parametresi
            r1 = self.a - t * (self.a / t_max)

            # Her ajan için konum güncelleme
            for i in range(pop_size):
                # Rastgele parametreler (vektörleştirilmiş: boyut başına bağımsız)
                r2 = rng.uniform(0, 2 * np.pi, size=n_features)
                r3 = rng.uniform(0, 2, size=n_features)
                r4 = rng.uniform(0, 1, size=n_features)

                w = population[i]
                diff = np.abs(r3 * p_best - w)  # |r3 * P_best - w_i|

                # Sinüs kolu (r4 < 0.5)
                sin_update = r1 * np.sin(r2) * diff
                # Kosinüs kolu (r4 >= 0.5)
                cos_update = r1 * np.cos(r2) * diff

                # Her boyut için uygun güncellemeyi seç
                population[i] = w + np.where(r4 < 0.5, sin_update, cos_update)

                # Sınır kontrolü (sınır yansıtma / kırpma)
                population[i] = np.clip(population[i], self.w_min, self.w_max)

            # ---------------------------------------------------------------
            # 4. Uygunluk değerlerini güncelle; yeni en iyi ajanı belirle
            # ---------------------------------------------------------------
            fitness = self._evaluate_population(population, X, y)
            current_best_idx = int(np.argmin(fitness))
            current_best_fitness = float(fitness[current_best_idx])

            if current_best_fitness < self.best_fitness_:
                self.best_fitness_ = current_best_fitness
                p_best = population[current_best_idx].copy()

            self.fitness_history_.append(self.best_fitness_)

            if self.verbose and (t % max(1, t_max // 10) == 0 or t == 1):
                print(
                    f"  [SCA] Iterasyon {t:>4}/{t_max}  |  "
                    f"En İyi MSE = {self.best_fitness_:.6f}"
                )

        # ---------------------------------------------------------------
        # 5. Sonuçları sakla
        # ---------------------------------------------------------------
        self.best_weights_ = p_best
        return self

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Optimize edilmiş ağırlıklarla duygu tahmini yap.

        Parametreler
        ------------
        X : np.ndarray, şekil (n_samples, 10)
            Tahmin yapılacak özellik matrisi.

        Döndürür
        --------
        y_pred : np.ndarray, şekil (n_samples,)
            Her örnek için ham duygu skoru (ŷ = X @ w).
            Sınıf etiketi için np.sign(y_pred) kullanılabilir.
        """
        if self.best_weights_ is None:
            raise RuntimeError(
                "Model henüz eğitilmedi. Önce fit() metodunu çağırın."
            )
        return _linear_score(X, self.best_weights_)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """
        Sürekli skoru {-1, 1} sınıf etiketine dönüştür.

        Döndürür
        --------
        labels : np.ndarray, şekil (n_samples,)
            +1 (Olumlu) veya -1 (Olumsuz) etiketleri.
        """
        scores = self.predict(X)
        # Skor = 0 ise olumlu (+1) kabul edilir
        return np.where(scores >= 0, 1, -1).astype(int)

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _evaluate_population(
        self, population: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Tüm popülasyon için MSE uygunluk değerlerini vektörleştirilmiş biçimde hesapla.

        population @ X.T  =>  şekil (pop_size, n_samples)
        Her satır bir ajanın tahminleri.
        """
        # (pop_size, n_samples) = (pop_size, n_features) @ (n_features, n_samples)
        all_preds = population @ X.T
        # Her ajan için MSE: (pop_size,)
        errors = all_preds - y  # yayımlama (broadcasting) ile y çıkar
        mse_values = np.mean(errors ** 2, axis=1)
        return mse_values

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Test seti üzerinde MSE değerini döndür.

        Döndürür
        --------
        mse : float
        """
        y_pred = self.predict(X)
        return _mse(y, y_pred)

    def feature_importance(self) -> dict[str, float]:
        """
        Optimize edilmiş mutlak ağırlıkları özellik adlarıyla eşleştir.

        Döndürür
        --------
        importance : dict
            {özellik_adı: |ağırlık|} sözlüğü, büyükten küçüğe sıralı.
        """
        if self.best_weights_ is None:
            raise RuntimeError("Model henüz eğitilmedi.")
        abs_weights = np.abs(self.best_weights_)
        importance = {
            name: float(w)
            for name, w in zip(self.FEATURE_NAMES, abs_weights)
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def __repr__(self) -> str:
        trained = self.best_weights_ is not None
        return (
            f"SCA_SentimentOptimizer("
            f"a={self.a}, w_min={self.w_min}, w_max={self.w_max}, "
            f"trained={trained})"
        )


# ---------------------------------------------------------------------------
# Test Bloğu
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 10,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    10 boyutlu sentetik özellik matrisi ve ikili {-1, +1} etiketleri üret.

    Gerçek uygulama notu:
        Gerçek veride X, her tweet için hesaplanmış 10 dilbilimsel metriği
        (bkz. FEATURE_NAMES) içeren, önceden normalize edilmiş bir matristir.

    Sentetik veri üretim mantığı:
        - X ~ Uniform[-1, 1] ile oluşturulan özellik matrisi
        - Gerçek ağırlıklar w_true rastgele seçilir
        - Ham skor = X @ w_true + gürültü
        - Etiket = sign(ham skor)  =>  {-1, +1}
    """
    rng = np.random.default_rng(random_state)

    # Özellik matrisi: her satır bir tweet, her sütun bir dilbilimsel metrik
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))

    # Gizli gerçek ağırlıklar (optimizörün bulmaya çalışacağı)
    w_true = rng.uniform(-2.0, 2.0, size=n_features)

    # Gürültülü skor ve ikili etiket
    noise = rng.normal(0, 0.3, size=n_samples)
    raw_scores = X @ w_true + noise
    y = np.sign(raw_scores)              # {-1.0, 0.0, +1.0}
    y = np.where(y == 0, 1.0, y)        # 0 değerleri +1'e çevir

    return X, y


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """İkili sınıflandırma doğruluğu."""
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    print("=" * 60)
    print("  SCA Tabanlı Duygu Analizi Optimizörü – Test Çalıştırması")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Sentetik veri üret
    # ------------------------------------------------------------------
    N_SAMPLES = 600
    N_FEATURES = 10   # 10 dilbilimsel kıstas
    TRAIN_SIZE = 480  # %80 eğitim

    X_all, y_all = _generate_synthetic_data(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=7
    )
    X_train, y_train = X_all[:TRAIN_SIZE], y_all[:TRAIN_SIZE]
    X_test, y_test = X_all[TRAIN_SIZE:], y_all[TRAIN_SIZE:]

    print(f"\nVeri Boyutu  : {X_all.shape}  (eğitim: {TRAIN_SIZE}, test: {N_SAMPLES - TRAIN_SIZE})")
    print(f"Özellik Sayısı: {N_FEATURES}")
    print(f"Sınıf Dağılımı: Olumlu={int((y_all == 1).sum())}, "
          f"Olumsuz={int((y_all == -1).sum())}\n")

    # ------------------------------------------------------------------
    # 2. Modeli eğit
    # ------------------------------------------------------------------
    model = SCA_SentimentOptimizer(
        a=2.0,
        w_min=-5.0,
        w_max=5.0,
        random_state=42,
        verbose=True,
    )

    print("SCA Optimizasyonu Başlıyor...\n")
    model.fit(X_train, y_train, epochs=100, pop_size=30)

    # ------------------------------------------------------------------
    # 3. Sonuçları değerlendir
    # ------------------------------------------------------------------
    train_mse = model.score(X_train, y_train)
    test_mse = model.score(X_test, y_test)

    y_train_pred = model.predict_class(X_train)
    y_test_pred = model.predict_class(X_test)

    train_acc = _accuracy(y_train.astype(int), y_train_pred)
    test_acc = _accuracy(y_test.astype(int), y_test_pred)

    print("\n" + "-" * 60)
    print("Eğitim Sonuçları:")
    print(f"  Eğitim MSE      : {train_mse:.6f}")
    print(f"  Test     MSE    : {test_mse:.6f}")
    print(f"  Eğitim Doğruluğu: {train_acc * 100:.2f}%")
    print(f"  Test  Doğruluğu : {test_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Optimize edilmiş ağırlıklar ve özellik önemi
    # ------------------------------------------------------------------
    print("\nOptimize Edilmiş Ağırlık Vektörü:")
    for name, w in zip(SCA_SentimentOptimizer.FEATURE_NAMES, model.best_weights_):
        print(f"  {name:<38}: {w:+.4f}")

    print("\nÖzellik Önemi (|ağırlık|, büyükten küçüğe):")
    for name, imp in model.feature_importance().items():
        bar = "█" * int(imp * 10)
        print(f"  {name:<38}: {imp:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("Test tamamlandı.")
    print("=" * 60)
