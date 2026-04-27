import os
import re
import nltk

# İlk çalıştırmada gerekli NLTK verilerini indir
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

STOPWORDS_EN = set(stopwords.words("english"))


def clean_gutenberg_header(text):
    """Project Gutenberg başlık ve sonlarını kaldırır."""
    
    # Başlık sonu işareti
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    # Son başlangıç işareti
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]
    
    # Başlığı atla
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            text = text[text.find("\n") + 1:]  # İşaret satırını da atla
            break
    
    # Sonu atla
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    
    return text.strip()


def clean_text(text):
    """Metni temizler: küçük harf, gereksiz boşluk, özel karakter."""
    text = text.lower()
    text = re.sub(r"\n+", " ", text)       # Satır sonlarını boşluğa çevir
    text = re.sub(r"\s+", " ", text)       # Fazla boşlukları temizle
    text = re.sub(r"[^a-z\s.,!?;:'\"()-]", "", text)  # Sadece harf ve noktalama
    return text.strip()


def split_into_chunks(text, chunk_size=500):
    """
    Metni cümle bazında chunk'lara böler.
    Her chunk yaklaşık chunk_size kelime içerir.
    Neden chunk? Modele uzun kitap vermek yerine
    küçük parçalar veririz — her parça bir örnek olur.
    """
    sentences = sent_tokenize(text)
    
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
    
    # Kalan cümleler
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def load_author_texts(data_dir="data/raw"):
    """
    data/raw/ altındaki tüm yazarların metinlerini yükler.
    Döndürür: [(chunk, yazar_adı), ...] listesi
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
            
            # Gutenberg başlığını temizle
            text = clean_gutenberg_header(raw_text)
            # Metni temizle
            text = clean_text(text)
            # Chunk'lara böl
            chunks = split_into_chunks(text, chunk_size=500)
            
            print(f"{author}/{filename}: {len(chunks)} chunk")
            
            for chunk in chunks:
                dataset.append((chunk, author, filename))
    
    return dataset


if __name__ == "__main__":
    dataset = load_author_texts()
    print(f"\nToplam: {len(dataset)} chunk")
    
    # Her yazardan kaç chunk var?
    from collections import Counter
    counts = Counter(author for _, author in dataset)
    for author, count in counts.items():
        print(f"  {author}: {count} chunk")