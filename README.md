# AI Yazar Tespiti — Stilometri Tabanlı

Metinlerin yazarını stilometri yöntemleriyle otomatik olarak tespit eden makine öğrenmesi projesi.

## Proje Hakkında

Stilometri; kelime seçimi, cümle uzunluğu, noktalama kullanımı ve karakter n-gramları gibi dilsel örüntüleri analiz ederek bir metnin yazarını belirleme bilimidir. Bu proje scikit-learn tabanlı klasik ML modelleri kullanır.

## Klasör Yapısı

```
ai-yazar-tespit/
├── data/
│   ├── raw/            # Ham metin dosyaları (yazar bazında alt klasörler)
│   └── processed/      # Özellik matrisleri ve etiketler
├── src/
│   ├── preprocessing.py   # Metin temizleme ve normalizasyon
│   ├── features.py        # Stilometrik özellik çıkarımı
│   ├── model.py           # Model eğitimi, değerlendirme ve kaydetme
│   └── utils.py           # Yardımcı fonksiyonlar
├── notebooks/
│   └── exploratory.ipynb  # Keşifsel veri analizi
├── tests/
│   └── test_features.py   # Birim testler
├── models/                # Eğitilmiş model dosyaları (.pkl)
├── reports/               # Grafik ve değerlendirme raporları
├── main.py                # Ana çalıştırma betiği
└── requirements.txt
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
python -m spacy download tr_core_news_sm   # Türkçe model (opsiyonel)
```

## Kullanım

```bash
# Veriyi işle ve özellikleri çıkar
python main.py --mode train --data data/raw

# Yeni bir metin için tahmin yap
python main.py --mode predict --text "Tahmin edilecek metin buraya."
```

## Özellikler (Features)

| Kategori | Özellikler |
|---|---|
| Leksikal | Ortalama kelime uzunluğu, kelime dağarcığı zenginliği, TTR |
| Sözdizimsel | Ortalama cümle uzunluğu, noktalama yoğunluğu |
| Karakter | Karakter n-gramları (bi/tri-gram) |
| Kelime | En sık kullanılan kelimeler, durma kelimesi oranı |

## Modeller

- Lojistik Regresyon
- Destek Vektör Makinesi (SVM)
- Rastgele Orman (Random Forest)

## Gereksinimler

Python 3.9+
