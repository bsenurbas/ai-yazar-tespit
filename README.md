# Uslup Madenciligi: Yazar Tespiti ve Yerel LLM Degerlendirmesi

Bu proje, edebi metinlerde yazar uslubunu tespit etmek ve yerel LLM modelleriyle uretilen metinleri stilometrik olarak degerlendirmek icin gelistirilmis bir metin madenciligi uygulamasidir.

Streamlit arayuzu uzerinden kullanici bir metin girer; sistem metnin hangi yazara daha yakin oldugunu tahmin eder, karar skorlarini gosterir, stilometrik metrikleri hesaplar ve yerel LLM uretimlerini kalite kontrol kurallariyla inceler.

## Kapsam

Projede iki yazar grubu vardir:

- Turkce yazarlar: Ahmet Rasim, Omer Seyfettin, Sabahattin Ali
- Yabanci yazarlar: Arthur Conan Doyle, Edgar Allan Poe, H. G. Wells

Turkce modelde TF-IDF karakter n-gram ve kelime n-gram ozellikleri birlikte kullanilmis, siniflandirici olarak LinearSVC tercih edilmistir. Degerlendirmede veri sizintisini azaltmak icin kitap bazli ayrim uygulanmistir.

## Temel Sonuclar

- Turkce production book-level split: %100.00
- Coklu book-level validation ortalamasi: %90.49
- Production egitim kumesi: 222 ornek
- Yazar basina dengeli ornek: 74
- Manuel gercek eser testi: 3/3 dogru
- Turkish-LLM-7B Q4 yerel uretim deneyi: 5/15 hedef yazar eslesmesi

## Klasor Yapisi

```text
ai-yazar-tespit/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/                 # Yabanci yazar ham metinleri
│   └── raw_turkish/         # Turkce yazar ham metinleri
├── models/
│   ├── tfidf_pipeline.pkl
│   ├── label_encoder.pkl
│   ├── turkish_tfidf_pipeline.pkl
│   ├── turkish_label_encoder.pkl
│   └── turkish_model_metadata.json
├── reports/
│   ├── turkish_author_style_profiles.json
│   ├── turkish_chunk_experiments.json
│   ├── turkish_confusion_matrix.png
│   ├── turkish_llm_7b_q4_long_results.csv
│   └── turkish_multi_book_validation.json
└── src/
    ├── build_author_style_profiles.py
    ├── data_collector.py
    ├── features.py
    ├── preprocessing.py
    ├── tfidf_model.py
    ├── turkish_features.py
    ├── turkish_model.py
    └── turkish_preprocessing.py
```

## Kurulum

Windows PowerShell icin:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
```

Projede daha once olusturulmus `venv` klasoru varsa yeni ortam kurmadan dogrudan etkinlestirilebilir:

```powershell
.\venv\Scripts\activate
```

## Arayuzu Calistirma

```powershell
streamlit run app.py
```

Komut calistiktan sonra Streamlit tarayicida yerel arayuzu acar. Arayuzde:

- Yazar tahmini yapilabilir.
- Turkce ve yabanci yazar gruplari arasinda gecis yapilabilir.
- Karar skorlari ve guven duzeyi incelenebilir.
- Stilometrik metrikler gorulebilir.
- Yerel LLM ile metin uretimi denenebilir.
- Uretilen metinler kalite kontrol tablosuyla degerlendirilebilir.

## Modeli Yeniden Egitme

Turkce modeli yeniden egitmek icin:

```powershell
python -m src.turkish_model
```

Bu komut:

- `data/raw_turkish` altindaki ham metinleri okur.
- Metinleri 100-200 kelimelik parcalara ayirir.
- Kitap bazli dogrulama yapar.
- Nihai production modelini `models/turkish_tfidf_pipeline.pkl` olarak kaydeder.
- Label encoder ve metadata dosyalarini gunceller.

Yazar stil profillerini yeniden uretmek icin:

```powershell
python -m src.build_author_style_profiles
```

Bu komut `reports/turkish_author_style_profiles.json` dosyasini olusturur.

## Yerel LLM Kullanimi

Arayuzdeki yerel uretim bolumu Ollama uzerinden calisir. Ollama kuruluysa ve model indirilmis durumdaysa arayuzden secilen modele prompt gonderilir.

Denemelerde kullanilan modeller:

- Gemma3 4B
- Qwen2.5 tabanli LoRA denemeleri
- Turkish-LLM-7B Q4

LLM uretimleri otomatik olarak basarili kabul edilmez. Uygulama; kelime sayisi, meta/aciklama varligi, bozuk karakter, tekrar orani ve hedef yazar eslesmesi gibi kalite kontrollerini ayrica gosterir.

## Kullanilan Yontemler

- TF-IDF
- Karakter n-gramlari
- Kelime n-gramlari
- LinearSVC
- Kitap bazli dogrulama
- Stilometrik metrikler
- Yerel LLM uretim kalite kontrolu

## Gereksinimler

- Python 3.9+
- Streamlit
- scikit-learn
- NumPy
- NLTK
- Matplotlib
- Seaborn

Tam paket listesi `requirements.txt` dosyasinda bulunur.
