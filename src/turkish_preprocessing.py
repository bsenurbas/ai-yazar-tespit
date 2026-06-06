import os
import re
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


# Türkçe stopword listesi
TURKISH_STOPWORDS = set([
    "ve", "ile", "de", "da", "ki", "bu", "bir", "o", "ben", "sen",
    "biz", "siz", "onlar", "için", "ama", "fakat", "lakin", "ancak",
    "ne", "ya", "hem", "veya", "yahut", "ise", "gibi", "kadar",
    "daha", "en", "çok", "az", "bile", "dahi", "hiç", "her", "bazı",
    "bütün", "tüm", "hep", "artık", "şimdi", "sonra", "önce", "yine",
    "tekrar", "sadece", "yalnız", "öyle", "böyle", "şöyle", "nasıl",
    "neden", "niçin", "nerede", "nereden", "nereye", "kim", "kimin",
    "hangi", "mi", "mu", "mı", "mü", "değil", "var", "yok", "olan",
    "oldu", "olur", "olan", "dedi", "diye", "demek", "demişti",
    "etti", "etmek", "etmiş", "olarak", "üzere", "göre", "karşı",
    "doğru", "beri", "itibaren", "rağmen", "başka", "diğer", "aynı"
])


def clean_turkish_text(text):
    """
    Türkçe metni temizler.
    Wiki markup, gereksiz karakterler ve boşlukları kaldırır.
    """
    # Wiki markup temizle
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)
    text = re.sub(r'={2,}[^=]+=+', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Özel karakterleri temizle
    text = re.sub(r'[🙝🙟★☆►◄→←]', '', text)
    text = re.sub(r'\*+', '', text)

    # Satır sonlarını düzenle
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Küçük harfe çevir (Türkçe'ye özgü: I→ı, İ→i)
    text = text.replace('I', 'ı').replace('İ', 'i')
    text = text.lower()

    # Sadece harf, rakam ve noktalama bırak
    text = re.sub(r'[^a-züşğıöçA-ZÜŞĞIÖÇa-züşğıöç\s.,!?;:\'\"-]', '', text)

    return text.strip()


def split_into_chunks_turkish(text, chunk_size=300):
    """
    Türkçe metni cümle bazında chunk'lara böler.
    chunk_size: hedef kelime sayısı
    """
    try:
        sentences = sent_tokenize(text, language='turkish')
    except Exception:
        # Turkish dil modeli yoksa noktalama ile böl
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        current_chunk.append(sentence)
        current_word_count += len(words)

        if current_word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_turkish_texts(data_dir="data/raw_turkish"):
    """
    data/raw_turkish/ altındaki tüm metinleri yükler.
    Döndürür: [(chunk, yazar, dosya_adi), ...]
    """
    dataset = []

    for author in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author)

        if not os.path.isdir(author_path):
            continue

        for filename in os.listdir(author_path):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(author_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Temizle
            text = clean_turkish_text(raw_text)

            # Chunk'lara böl
            chunks = split_into_chunks_turkish(text, chunk_size=300)

            # Çok kısa chunk'ları filtrele
            chunks = [c for c in chunks if len(c.split()) >= 150]

            print(f"{author}/{filename}: {len(chunks)} chunk")

            for chunk in chunks:
                dataset.append((chunk, author, filename))

    return dataset


if __name__ == "__main__":
    print("Türkçe metinler yükleniyor...\n")
    dataset = load_turkish_texts()

    print(f"\nToplam: {len(dataset)} chunk")

    from collections import Counter
    counts = Counter(author for _, author, _ in dataset)
    for author, count in sorted(counts.items()):
        print(f"  {author}: {count} chunk")