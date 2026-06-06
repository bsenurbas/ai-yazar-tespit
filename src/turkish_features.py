import re
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Türkçe stopword listesi
TURKISH_STOPWORDS = set([
    "ve", "ile", "de", "da", "ki", "bu", "bir", "o", "ben", "sen",
    "biz", "siz", "onlar", "için", "ama", "fakat", "lakin", "ancak",
    "ne", "ya", "hem", "veya", "yahut", "ise", "gibi", "kadar",
    "daha", "en", "çok", "az", "bile", "dahi", "hiç", "her", "bazı",
    "bütün", "tüm", "hep", "artık", "şimdi", "sonra", "önce", "yine",
    "tekrar", "sadece", "yalnız", "öyle", "böyle", "şöyle", "nasıl",
    "neden", "niçin", "nerede", "nereden", "nereye", "kim", "kimin",
    "hangi", "mi", "mu", "mı", "mü", "değil", "var", "yok",
    "olan", "oldu", "olur", "dedi", "diye", "demek",
    "etti", "etmek", "etmiş", "olarak", "üzere", "göre", "karşı",
    "doğru", "beri", "itibaren", "rağmen", "başka", "diğer", "aynı"
])

# Türkçe fiil ekleri (sonekleri)
TURKISH_VERB_SUFFIXES = [
    "di", "dı", "du", "dü", "ti", "tı", "tu", "tü",
    "miş", "mış", "muş", "müş", "meli", "malı",
    "yor", "iyor", "uyor", "üyor",
    "ecek", "acak", "mak", "mek",
    "arak", "erek", "ince", "unca",
]


def avg_word_length(words):
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def avg_sentence_length(sentences):
    if not sentences:
        return 0
    return np.mean([len(s.split()) for s in sentences])


def type_token_ratio(words):
    if not words:
        return 0
    return len(set(words)) / len(words)


def stopword_ratio(words):
    if not words:
        return 0
    sw_count = sum(1 for w in words if w in TURKISH_STOPWORDS)
    return sw_count / len(words)


def turkish_suffix_ratio(words):
    """
    Türkçe'ye özgü özellik: fiil eki kullanan kelime oranı.
    Yazarın fiil yoğunluğunu yansıtır.
    """
    if not words:
        return 0
    suffix_count = sum(
        1 for w in words
        if any(w.endswith(suf) for suf in TURKISH_VERB_SUFFIXES)
    )
    return suffix_count / len(words)


def avg_word_length_chars(words):
    """Ortalama kelime uzunluğu karakter sayısı olarak."""
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def punctuation_features(text):
    total_chars = len(text) if text else 1
    return {
        "comma_rate": text.count(",") / total_chars,
        "semicolon_rate": text.count(";") / total_chars,
        "exclamation_rate": text.count("!") / total_chars,
        "question_rate": text.count("?") / total_chars,
        "dash_rate": text.count("—") / total_chars,
        "ellipsis_rate": text.count("...") / total_chars,
    }


def char_ngram_features(text, n=3, top_k=30):
    """
    Türkçe karakter trigramları.
    Türkçe için top_k=30 daha iyi — eklemeli yapı çok çeşitli trigram üretir.
    """
    clean = re.sub(r"\s+", "", text.lower())
    ngrams = [clean[i:i+n] for i in range(len(clean) - n + 1)]

    if not ngrams:
        return {}

    counts = Counter(ngrams)
    total = sum(counts.values())

    return {
        f"char_{n}gram_{ng}": cnt / total
        for ng, cnt in counts.most_common(top_k)
    }


def extract_turkish_features(text):
    """
    Tek bir chunk için tüm Türkçe stilometrik özellikleri çıkarır.
    """
    try:
        sentences = sent_tokenize(text, language='turkish')
    except Exception:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    words = text.lower().split()
    alpha_words = [w for w in words if w.isalpha()]

    features = {}

    # 1. Leksikal özellikler
    features["avg_word_length"] = avg_word_length(alpha_words)
    features["avg_sentence_length"] = avg_sentence_length(sentences)
    features["type_token_ratio"] = type_token_ratio(alpha_words)
    features["stopword_ratio"] = stopword_ratio(alpha_words)
    features["word_count"] = len(alpha_words)

    # 2. Türkçe'ye özgü özellikler
    features["verb_suffix_ratio"] = turkish_suffix_ratio(alpha_words)

    # 3. Noktalama özellikleri
    features.update(punctuation_features(text))

    # 4. Karakter trigramları (Türkçe için top_k=30)
    features.update(char_ngram_features(text, n=3, top_k=30))

    return features


class TurkishStylometricTransformer(BaseEstimator, TransformerMixin):
    """Sklearn pipeline icin sabit boyutlu Turkce stilometrik ozellikler."""

    FEATURE_NAMES = [
        "avg_word_length",
        "avg_sentence_length",
        "type_token_ratio",
        "stopword_ratio",
        "word_count",
        "verb_suffix_ratio",
        "comma_rate",
        "semicolon_rate",
        "exclamation_rate",
        "question_rate",
        "dash_rate",
        "ellipsis_rate",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            features = extract_turkish_features(text)
            rows.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])
        return np.asarray(rows, dtype=float)


def build_turkish_feature_matrix(dataset):
    """
    Tüm dataset için özellik matrisini oluşturur.
    dataset: [(chunk, yazar, dosya_adi), ...]
    """
    all_features = []
    labels = []
    groups = []

    for chunk, author, filename in dataset:
        feat = extract_turkish_features(chunk)
        all_features.append(feat)
        labels.append(author)
        groups.append(filename)

    # Tüm feature isimlerini topla
    all_keys = set()
    for f in all_features:
        all_keys.update(f.keys())
    feature_names = sorted(all_keys)

    X = np.array([
        [feat.get(k, 0) for k in feature_names]
        for feat in all_features
    ])

    return X, labels, groups, feature_names


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.turkish_preprocessing import load_turkish_texts
    from src.utils import balance_dataset

    print("Metinler yükleniyor...")
    dataset = load_turkish_texts()

    print("Veri dengeleniyor...")
    # balance_dataset (chunk, author) tuple bekliyor
    dataset_for_balance = [(d[0], d[1]) for d in dataset]
    balanced = balance_dataset(dataset_for_balance)

    # filename bilgisini geri ekle
    filename_map = {(d[0], d[1]): d[2] for d in dataset}
    balanced_full = []
    for chunk, author in balanced:
        filename = filename_map.get((chunk, author), "unknown")
        balanced_full.append((chunk, author, filename))

    print(f"Dengeli dataset: {len(balanced_full)} chunk")

    print("Özellikler çıkarılıyor...")
    X, y, groups, feature_names = build_turkish_feature_matrix(balanced_full)

    print(f"Özellik matrisi: {X.shape}")
    print(f"Özellik sayısı: {len(feature_names)}")

    from collections import Counter
    counts = Counter(y)
    for author, count in sorted(counts.items()):
        print(f"  {author}: {count} chunk")
