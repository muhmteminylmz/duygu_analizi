"""
sca_sentiment_optimizer.py
==========================
Sine-Cosine Algorithm (SCA) tabanlı Duygu Analizi Ağırlık Optimizörü
3 Sınıflı Versiyon: Olumlu (+1), Nötr (0), Olumsuz (-1)

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
    Hedef etiketler: -1.0, 0.0, 1.0

Sınıflandırma Eşikleri (predict_class):
    Skor >= 0.25  → Olumlu (+1)
    Skor <= -0.25 → Olumsuz (-1)
    -0.25 < Skor < 0.25 → Nötr (0)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
        Sürekli skoru {-1, 0, +1} sınıf etiketine dönüştür (3 sınıflı).

        Eşik Kuralları:
            Skor >= 0.25  → Olumlu (+1)
            Skor <= -0.25 → Olumsuz (-1)
            -0.25 < Skor < 0.25 → Nötr (0)

        Döndürür
        --------
        labels : np.ndarray, şekil (n_samples,)
            +1 (Olumlu), 0 (Nötr) veya -1 (Olumsuz) etiketleri.
        """
        scores = self.predict(X)
        labels = np.zeros(len(scores), dtype=int)
        labels[scores >= 0.25] = 1
        labels[scores <= -0.25] = -1
        return labels

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

    def plot_results(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str | None = None,
    ) -> None:
        """
        Eğitim sonuçlarını 3 farklı grafik ile görselleştirir.

        Grafik 1: SCA Yakınsama Eğrisi – iterasyon bazında en iyi MSE.
        Grafik 2: Özellik Önemi        – mutlak ağırlıkların yatay bar grafiği.
        Grafik 3: Karmaşıklık Matrisi  – test seti için Heatmap (Olumlu/Nötr/Olumsuz).

        Parametreler
        ------------
        X_test : np.ndarray
            Test özellik matrisi.
        y_test : np.ndarray
            Gerçek test etiketleri ({-1, 0, 1}).
        save_path : str veya None
            Grafiklerin kaydedileceği dosya yolu (örn. "results.png").
            None ise grafik ekranda gösterilir.
        """
        if self.best_weights_ is None:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() çağırın.")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("SCA Duygu Analizi – Sonuç Görselleştirmeleri", fontsize=15, fontweight="bold")

        # ------------------------------------------------------------------
        # Grafik 1: SCA Yakınsama Eğrisi
        # ------------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(range(len(self.fitness_history_)), self.fitness_history_,
                 color="steelblue", linewidth=2)
        ax1.set_title("SCA Yakınsama Eğrisi", fontsize=13)
        ax1.set_xlabel("İterasyon")
        ax1.set_ylabel("En İyi MSE")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.fill_between(range(len(self.fitness_history_)), self.fitness_history_,
                         alpha=0.15, color="steelblue")

        # ------------------------------------------------------------------
        # Grafik 2: Özellik Önemi (yatay bar grafiği, büyükten küçüğe)
        # ------------------------------------------------------------------
        ax2 = axes[1]
        importance = self.feature_importance()  # zaten büyükten küçüğe sıralı
        names = list(importance.keys())
        values = list(importance.values())
        colors = sns.color_palette("viridis", len(names))
        bars = ax2.barh(names[::-1], values[::-1], color=colors)
        ax2.set_title("Özellik Önemi (|Ağırlık|)", fontsize=13)
        ax2.set_xlabel("|Ağırlık|")
        ax2.grid(True, axis="x", linestyle="--", alpha=0.6)
        # Değer etiketleri
        for bar, val in zip(bars, values[::-1]):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)

        # ------------------------------------------------------------------
        # Grafik 3: Karmaşıklık Matrisi (Confusion Matrix)
        # ------------------------------------------------------------------
        ax3 = axes[2]
        y_pred = self.predict_class(X_test)
        labels_order = [-1, 0, 1]
        label_names = ["Olumsuz (-1)", "Nötr (0)", "Olumlu (+1)"]
        cm = confusion_matrix(y_test.astype(int), y_pred, labels=labels_order)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax3,
            linewidths=0.5,
        )
        ax3.set_title("Karmaşıklık Matrisi (Test Seti)", fontsize=13)
        ax3.set_xlabel("Tahmin Edilen Sınıf")
        ax3.set_ylabel("Gerçek Sınıf")
        ax3.tick_params(axis="x", rotation=15)
        ax3.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Grafikler kaydedildi: {save_path}")
        else:
            plt.show()

    def __repr__(self) -> str:
        trained = self.best_weights_ is not None
        return (
            f"SCA_SentimentOptimizer("
            f"a={self.a}, w_min={self.w_min}, w_max={self.w_max}, "
            f"trained={trained})"
        )


# ---------------------------------------------------------------------------
# Gerçek Veri – HuggingFace Veri Seti Yükleme & Özellik Çıkarımı
# ---------------------------------------------------------------------------

# Türkçe kelime listeleri (özellik çıkarımında kullanılır)
_NEGATION_WORDS = {
    "değil", "yok", "hayır", "olmaz", "olmadı", "etmez", "istemiyorum",
    "istemez", "bilmez", "bilmiyorum", "hiç", "hiçbir", "hiçbiri",
}
_INTENSIFIERS = {
    "çok", "aşırı", "fazla", "son derece", "kesinlikle", "tam", "gerçekten",
    "müthiş", "inanılmaz", "fevkalade", "son", "pek", "gayet", "oldukça",
    "epey", "bayağı", "deli gibi", "bir hayli",
}
_POSITIVE_WORDS = {
    "güzel", "harika", "muhteşem", "mükemmel", "iyi", "süper", "sevdim",
    "beğendim", "başarılı", "kaliteli", "memnun", "mutlu", "seviyorum",
    "sevgili", "tatlı", "hoş", "şahane", "efsane", "bravo", "tebrikler",
    "teşekkür", "teşekkürler", "sağ ol", "sağolun", "iyilik", "kusursuz",
    "enfes", "nefis",
}
_NEGATIVE_WORDS = {
    "kötü", "berbat", "rezalet", "korkunç", "iğrenç", "nefret", "sinirli",
    "üzgün", "mutsuz", "şikayet", "sorun", "problem", "hata", "yanlış",
    "kötüydü", "beğenmedim", "sevmedim", "şikayetçi", "dandik", "çöp",
    "rezil", "pişman", "üzücü", "acı", "can sıkıcı", "sıkıcı",
}
_CONTRAST_CONJUNCTIONS = {"ama", "fakat", "lakin", "ancak", "ne var ki", "oysa", "yalnız", "bununla birlikte"}
_ADJECTIVE_ADVERBS = {
    "güzel", "kötü", "büyük", "küçük", "hızlı", "yavaş", "iyi", "güçlü",
    "zayıf", "uzun", "kısa", "eski", "yeni", "açık", "kapalı", "sıcak",
    "soğuk", "kolay", "zor", "doğru", "yanlış", "dolu", "boş", "hemen",
    "şimdi", "çabuk", "yine", "tekrar", "sadece", "bile", "dahi",
}


def _extract_features(text: str) -> np.ndarray:
    """
    Ham Türkçe metinden 10 dilbilimsel özelliği çıkarır ve normalize eder.

    Özellikler (FEATURE_NAMES ile aynı sırada):
        1. Değilleme (Negation)
        2. Yoğunlaştırıcılar (Intensifiers)
        3. Büyük Harf Oranı
        4. Noktalama Yoğunluğu
        5. Kelime Çeşitliliği (TTR)
        6. Fiil Zamanı (Tense)
        7. Öznellik Skoru (Subjectivity)
        8. Sıfat ve Zarf Oranı
        9. İroni Tespiti (Irony)
       10. Zıtlık Bağlaçları (Contrast)
    """
    if not text or not text.strip():
        return np.zeros(10)

    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)
    n_tokens = max(len(tokens), 1)

    # 1. Değilleme: olumsuzluk kelimesi oranı + Türkçe olumsuz ekler (-me/-ma)
    neg_count = sum(1 for t in tokens if t in _NEGATION_WORDS)
    neg_suffix = len(re.findall(r"\b\w+(?:me|ma|mez|maz|miyor|mıyor|muyor|müyor)\b", text_lower))
    negation = min((neg_count + neg_suffix) / n_tokens, 1.0)

    # 2. Yoğunlaştırıcılar: şiddet kelimesi oranı
    intensifier = min(sum(1 for t in tokens if t in _INTENSIFIERS) / n_tokens, 1.0)

    # 3. Büyük Harf Oranı: tamamen büyük harfli kelimeler
    raw_tokens = re.findall(r"\w+", text)
    upper_ratio = sum(1 for t in raw_tokens if t.isupper() and len(t) > 1) / max(len(raw_tokens), 1)

    # 4. Noktalama Yoğunluğu: ünlem ve soru işareti oranı
    punct_count = text.count("!") + text.count("?")
    punct_density = min(punct_count / max(len(text), 1) * 10, 1.0)

    # 5. Kelime Çeşitliliği (TTR): benzersiz kelime / toplam kelime
    ttr = len(set(tokens)) / n_tokens

    # 6. Fiil Zamanı: geçmiş/gelecek zaman eki oranı
    past_future = len(re.findall(
        r"\b\w+(?:dı|di|du|dü|tı|ti|tu|tü|acak|ecek|acağ|eceğ)\b", text_lower
    ))
    tense = min(past_future / n_tokens, 1.0)

    # 7. Öznellik Skoru: duygu kelimesi yoğunluğu
    sentiment_words = sum(1 for t in tokens if t in _POSITIVE_WORDS or t in _NEGATIVE_WORDS)
    subjectivity = min(sentiment_words / n_tokens, 1.0)

    # 8. Sıfat ve Zarf Oranı
    adj_adv_ratio = min(sum(1 for t in tokens if t in _ADJECTIVE_ADVERBS) / n_tokens, 1.0)

    # 9. İroni Tespiti: olumlu VE olumsuz kelime eş-bulunması (normalize)
    has_pos = any(t in _POSITIVE_WORDS for t in tokens)
    has_neg = any(t in _NEGATIVE_WORDS for t in tokens)
    irony = 1.0 if (has_pos and has_neg) else 0.0

    # 10. Zıtlık Bağlaçları: "ama/fakat/lakin" gibi bağlaç varlığı
    contrast = 1.0 if any(conj in text_lower for conj in _CONTRAST_CONJUNCTIONS) else 0.0

    return np.array([
        negation, intensifier, upper_ratio, punct_density, ttr,
        tense, subjectivity, adj_adv_ratio, irony, contrast,
    ], dtype=float)


def _load_huggingface_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    HuggingFace'den Türkçe duygu analizi veri setini indir ve
    (X, y, texts) üçlüsüne dönüştür.

    Kaynak: winvoker/turkish-sentiment-analysis-dataset
    Etiket dönüşümü:
        positive → +1.0 (Olumlu)
        negative → -1.0 (Olumsuz)
        notr / neutral → 0.0 (Nötr)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HuggingFace 'datasets' kütüphanesi bulunamadı. "
            "Kurmak için: pip install datasets"
        ) from exc

    print("HuggingFace veri seti indiriliyor: winvoker/turkish-sentiment-analysis-dataset ...")
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")

    # train/test split'lerini birleştir (varsa)
    splits = list(ds.keys())
    all_texts, all_labels = [], []
    for split in splits:
        split_data = ds[split]
        # Olası sütun adlarını otomatik tespit et
        text_col = next((c for c in split_data.column_names if "text" in c.lower() or "sentence" in c.lower()), None)
        label_col = next((c for c in split_data.column_names if "label" in c.lower() or "sentiment" in c.lower()), None)
        if text_col is None or label_col is None:
            raise ValueError(
                f"Veri setindeki sütun adları tanınamadı: {split_data.column_names}. "
                "Beklenen sütun adları 'text'/'sentence' ve 'label'/'sentiment' içermelidir."
            )

        all_texts.extend(split_data[text_col])
        raw_labels = split_data[label_col]
        for lbl in raw_labels:
            lbl_str = str(lbl).lower().strip()
            if lbl_str in ("1", "positive", "pos", "olumlu"):
                all_labels.append(1.0)
            elif lbl_str in ("0", "negative", "neg", "olumsuz"):
                all_labels.append(-1.0)
            elif lbl_str in ("notr", "nötr", "neutral", "ntr"):
                # Nötr sınıf: 0.0
                all_labels.append(0.0)
            else:
                # Sayısal: > 0 → +1, < 0 → -1, = 0 → 0 (nötr)
                try:
                    val = float(lbl_str)
                    if val > 0:
                        all_labels.append(1.0)
                    elif val < 0:
                        all_labels.append(-1.0)
                    else:
                        all_labels.append(0.0)
                except ValueError:
                    print(f"  UYARI: Bilinmeyen etiket formatı '{lbl}' varsayılan olarak +1 (Olumlu) kabul edildi.")
                    all_labels.append(1.0)

    print(f"Toplam {len(all_texts)} örnek yüklendi. Özellikler çıkarılıyor...")

    X = np.array([_extract_features(t) for t in all_texts], dtype=float)
    y = np.array(all_labels, dtype=float)

    return X, y, all_texts


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """3 sınıflı sınıflandırma doğruluğu."""
    return float(np.mean(y_true == y_pred))


def _label_name(label: int) -> str:
    """Sayısal etiketi Türkçe sınıf adına çevirir."""
    return {1: "Olumlu", 0: "Nötr", -1: "Olumsuz"}.get(label, str(label))


if __name__ == "__main__":
    print("=" * 60)
    print("  SCA Tabanlı 3-Sınıflı Duygu Analizi Optimizörü – Gerçek Veri")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Gerçek veri setini yükle ve özellikleri çıkar
    # ------------------------------------------------------------------
    X_all, y_all, texts_all = _load_huggingface_dataset()

    N_SAMPLES = len(y_all)
    TRAIN_SIZE = int(N_SAMPLES * 0.80)

    # Karıştır (shuffle) ve böl
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(N_SAMPLES)
    X_all, y_all = X_all[perm], y_all[perm]
    texts_all = [texts_all[i] for i in perm]

    X_train, y_train = X_all[:TRAIN_SIZE], y_all[:TRAIN_SIZE]
    X_test, y_test = X_all[TRAIN_SIZE:], y_all[TRAIN_SIZE:]
    texts_test = texts_all[TRAIN_SIZE:]

    print(f"\nVeri Boyutu  : {X_all.shape}  (eğitim: {TRAIN_SIZE}, test: {N_SAMPLES - TRAIN_SIZE})")
    print(f"Özellik Sayısı: {X_all.shape[1]}")
    print(
        f"Sınıf Dağılımı: Olumlu={int((y_all == 1).sum())}, "
        f"Nötr={int((y_all == 0).sum())}, "
        f"Olumsuz={int((y_all == -1).sum())}\n"
    )

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

    # ------------------------------------------------------------------
    # 5. Görselleştirme (3 grafik)
    # ------------------------------------------------------------------
    print("\nGrafikler oluşturuluyor...")
    model.plot_results(X_test, y_test)

    # ------------------------------------------------------------------
    # 6. Gerçek Örnekler Üzerinde Demo (10 örnek)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Demo: Rastgele 10 Örnek Üzerinde Model Performansı")
    print("=" * 60)

    # Test setinden rastgele 10 örnek seç
    rng_demo = np.random.default_rng(0)
    demo_idx = rng_demo.choice(len(texts_test), size=min(10, len(texts_test)), replace=False)

    demo_texts = [texts_test[i] for i in demo_idx]
    demo_X = X_test[demo_idx]
    demo_y_true = y_test[demo_idx].astype(int)
    demo_scores = model.predict(demo_X)
    demo_y_pred = model.predict_class(demo_X)

    # Tablo başlığı
    col_text = 52
    col_gercek = 14
    col_tahmin = 20
    col_skor = 14
    sep = "-" * (col_text + col_gercek + col_tahmin + col_skor + 5)
    header = (
        f"{'Metin (ilk 50 karakter)':<{col_text}} "
        f"{'Gerçek Sınıf':<{col_gercek}} "
        f"{'Tahmin Edilen Sınıf':<{col_tahmin}} "
        f"{'Ham Skor (SCA)':<{col_skor}}"
    )
    print(header)
    print(sep)
    for text, true_lbl, pred_lbl, score in zip(demo_texts, demo_y_true, demo_y_pred, demo_scores):
        text_short = (text[:49] + "…") if len(text) > 50 else text
        true_name = _label_name(int(true_lbl))
        pred_name = _label_name(int(pred_lbl))
        print(
            f"{text_short:<{col_text}} "
            f"{true_name:<{col_gercek}} "
            f"{pred_name:<{col_tahmin}} "
            f"{score:+.4f}"
        )
    print(sep)

    print("\n" + "=" * 60)
    print("Test tamamlandı.")
    print("=" * 60)
