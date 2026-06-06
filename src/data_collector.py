import urllib.request
import os
import time

# Project Gutenberg'den indirilecek kitaplar
# Format: (yazar_adı, eser_adı, gutenberg_url)
BOOKS = [
    # Edgar Allan Poe
    ("poe", "tales_of_mystery", "https://www.gutenberg.org/cache/epub/2147/pg2147.txt"),
    
    # Arthur Conan Doyle
    ("doyle", "adventures_of_sherlock_holmes", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
    ("doyle", "hound_of_baskervilles", "https://www.gutenberg.org/cache/epub/2852/pg2852.txt"),
    
    # H.G. Wells
    ("wells", "the_time_machine", "https://www.gutenberg.org/cache/epub/35/pg35.txt"),
    ("wells", "the_war_of_the_worlds", "https://www.gutenberg.org/cache/epub/36/pg36.txt"),

    # Wells için ek kitaplar
    ("wells", "the_invisible_man", "https://www.gutenberg.org/cache/epub/5230/pg5230.txt"),
    ("wells", "the_island_of_doctor_moreau", "https://www.gutenberg.org/cache/epub/159/pg159.txt"),
    ("wells", "the_first_men_in_the_moon", "https://www.gutenberg.org/cache/epub/1013/pg1013.txt"),

    # Doyle için ek kitap
    ("doyle", "memoirs_of_sherlock_holmes", "https://www.gutenberg.org/cache/epub/834/pg834.txt"),

    # Poe için ek eserler
    ("poe", "the_works_of_poe_vol1", "https://www.gutenberg.org/cache/epub/2148/pg2148.txt"),
    ("poe", "the_works_of_poe_vol3", "https://www.gutenberg.org/cache/epub/2150/pg2150.txt"),
]

def download_books(output_dir="data/raw"):
    """Kitapları Project Gutenberg'den indirir."""
    
    for author, title, url in BOOKS:
        # Yazar klasörünü oluştur
        author_dir = os.path.join(output_dir, author)
        os.makedirs(author_dir, exist_ok=True)
        
        filepath = os.path.join(author_dir, f"{title}.txt")
        
        # Daha önce indirildiyse atla
        if os.path.exists(filepath):
            print(f"Zaten mevcut: {author}/{title}")
            continue
        
        print(f"İndiriliyor: {author}/{title}...")
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8", errors="ignore")
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✓ İndirildi: {author}/{title}")
            time.sleep(2)  # Gutenberg'e saygılı ol, hızlı istek atma
            
        except Exception as e:
            print(f"✗ Hata: {author}/{title} — {e}")

if __name__ == "__main__":
    download_books()
    print("\nTüm indirmeler tamamlandı!")