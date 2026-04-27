import re
import string
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

STOPWORDS_EN = set(stopwords.words("english"))


def avg_word_length(words):
    """Ortalama kelime uzunluğu — yazarın kelime tercihini yansıtır."""
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def avg_sentence_length(sentences):
    """Ortalama cümle uzunluğu (kelime sayısı) — yazarın ritmi."""
    if not sentences:
        return 0
    lengths = [len(s.split()) for s in sentences]
    return np.mean(lengths)


def type_token_ratio(words):
    """
    TTR = benzersiz kelime sayısı / toplam kelime sayısı
    Yüksek TTR → zengin kelime dağarcığı
    """
    if not words:
        return 0
    return len(set(words)) / len(words)


def stopword_ratio(words):
    """
    Durma kelimesi oranı (the, a, an, is...)
    Yazarlar bu kelimeleri farklı sıklıkta kullanır.
    """
    if not words:
        return 0
    sw_count = sum(1 for w in words if w in STOPWORDS_EN)
    return sw_count / len(words)


def punctuation_features(text):
    """
    Noktalama işareti kullanım oranları.
    Poe çok virgül ve noktalı virgül kullanır mesela.
    """
    total_chars = len(text) if text else 1
    
    return {
        "comma_rate": text.count(",") / total_chars,
        "semicolon_rate": text.count(";") / total_chars,
        "exclamation_rate": text.count("!") / total_chars,
        "question_rate": text.count("?") / total_chars,
        "dash_rate": text.count("-") / total_chars,
    }


def char_ngram_features(text, n=3, top_k=20):
    """
    Karakter n-gramları — yazarın en çok kullandığı harf dizileri.
    Örnek trigramlar: 'the', 'ing', 'ion', 'tion'
    Bu özellik stilometride çok güçlüdür.
    """
    # Boşlukları kaldır, sadece harf
    clean = re.sub(r"\s+", "", text.lower())
    
    ngrams = [clean[i:i+n] for i in range(len(clean) - n + 1)]
    
    if not ngrams:
        return {}
    
    counts = Counter(ngrams)
    total = sum(counts.values())
    
    # En sık top_k n-gramı döndür (frekans olarak)
    return {f"char_{n}gram_{ng}": cnt / total 
            for ng, cnt in counts.most_common(top_k)}


def extract_features(text):
    """
    Tek bir chunk için tüm özellikleri çıkarır.
    Döndürür: özellik sözlüğü (dict)
    """
    # Tokenize
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    # Sadece alfabetik kelimeleri al
    alpha_words = [w for w in words if w.isalpha()]
    
    features = {}
    
    # 1. Leksikal özellikler
    features["avg_word_length"] = avg_word_length(alpha_words)
    features["avg_sentence_length"] = avg_sentence_length(sentences)
    features["type_token_ratio"] = type_token_ratio(alpha_words)
    features["stopword_ratio"] = stopword_ratio(alpha_words)
    features["word_count"] = len(alpha_words)
    
    # 2. Noktalama özellikleri
    features.update(punctuation_features(text))
    
    # 3. Karakter trigramları (en güçlü özellik)
    features.update(char_ngram_features(text, n=3, top_k=20))
    
    return features


def build_feature_matrix(dataset):
    """
    Tüm dataset için özellik matrisini oluşturur.
    dataset: [(chunk, yazar), ...] listesi
    Döndürür: (X, y, feature_names)
      X → numpy array, her satır bir chunk
      y → yazar etiketleri listesi
    """
    all_features = []
    labels = []
    
    for chunk, author in dataset:
        feat = extract_features(chunk)
        all_features.append(feat)
        labels.append(author)
    
    # Tüm feature isimlerini topla
    all_keys = set()
    for f in all_features:
        all_keys.update(f.keys())
    feature_names = sorted(all_keys)
    
    # Her chunk için tam vektör oluştur (eksik özellikler 0)
    X = np.array([
        [feat.get(k, 0) for k in feature_names]
        for feat in all_features
    ])
    
    return X, labels, feature_names


if __name__ == "__main__":
    from preprocessing import load_author_texts
    
    print("Metinler yükleniyor...")
    dataset = load_author_texts()
    
    print("Özellikler çıkarılıyor...")
    X, y, feature_names = build_feature_matrix(dataset)
    
    print(f"\nÖzellik matrisi boyutu: {X.shape}")
    print(f"Özellik sayısı: {len(feature_names)}")
    print(f"Örnek özellikler: {feature_names[:10]}")