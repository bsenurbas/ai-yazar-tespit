import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from itertools import product

RANDOM_STATE = 42
MIN_BOOKS_PER_AUTHOR = 2

RAW_DIR = Path("data/raw_turkish")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_PATH = MODEL_DIR / "turkish_tfidf_pipeline.pkl"
ENCODER_PATH = MODEL_DIR / "turkish_label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "turkish_model_metadata.json"
CONFUSION_PATH = REPORT_DIR / "turkish_confusion_matrix.png"

MIN_WORDS = 100
MAX_WORDS = 200
MAX_RECORDS_PER_AUTHOR = 120

MIN_TRAIN_RECORDS_PER_AUTHOR_FOR_VALIDATION = 30

EXPERIMENT_CONFIGS = [
    {"name": "short_80_160", "min_words": 80, "max_words": 160},
    {"name": "current_90_180", "min_words": 90, "max_words": 180},
    {"name": "medium_100_200", "min_words": 100, "max_words": 200},
    {"name": "long_120_220", "min_words": 120, "max_words": 220},
]

BAD_PATTERNS = [
    "filn",
    "demit",
    "olrn",
    "giyilr",
    "düklc",
    "saliaya",
    "malış",
    " rnis",
    " giin",
    " siiz",
    "bagr",
    "takibetti",
    "sıkid",
    "haydaı",
    "hatıralan",
    "hayalirole",
    " ber ",
    "h' ",
    "nekadar",
    "dü ",
    "p.er",
    "s du",
    "olmı",
    "□",
    "■",
    "�",
]


def normalize_text(text):
    return " ".join(text.split()).strip()


def clean_text_preserve_case(text):
    """Wiki/OCR artıklarını temizler; büyük/küçük harf bilgisini korur."""
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = re.sub(r"\[\[Kategori:[^\]]*\]\]", " ", text)
    text = re.sub(r"\[\[Dosya:[^\]]*\]\]", " ", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = text.replace("I", "ı")
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s.,!?;:'\"()\-]", " ", text)
    return normalize_text(text)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return [part.strip() for part in parts if len(part.split()) >= 3]


def has_bad_text(text):
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return True
    if any(pattern in lowered for pattern in BAD_PATTERNS):
        return True
    if not stripped[0].isupper():
        return True
    if lowered.count("...") > 2:
        return True
    if " . ." in lowered or " , ," in lowered:
        return True
    if re.search(r"\b\w+'\s+\w+", lowered):
        return True
    if re.search(r"[a-zçğıöşü]S\s+[a-zçğıöşü]", stripped):
        return True

    return False


def make_sentence_chunks(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if sentence_words > max_words:
            continue

        if current and current_words + sentence_words > max_words:
            chunk = normalize_text(" ".join(current))
            wc = len(chunk.split())
            if min_words <= wc <= max_words and not has_bad_text(chunk):
                chunks.append(chunk)
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunk = normalize_text(" ".join(current))
        wc = len(chunk.split())
        if min_words <= wc <= max_words and not has_bad_text(chunk):
            chunks.append(chunk)

    return chunks


def load_sentence_records(data_dir=RAW_DIR, min_words=MIN_WORDS, max_words=MAX_WORDS):
    dataset = []

    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue

        author = author_dir.name
        for path in sorted(author_dir.glob("*.txt")):
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            clean_text = clean_text_preserve_case(raw_text)
            chunks = make_sentence_chunks(
                clean_text,
                min_words=min_words,
                max_words=max_words,
            )
            print(f"{author}/{path.name}: {len(chunks)} chunk")

            for chunk in chunks:
                dataset.append((chunk, author, path.name))

    return dataset


def filter_authors_with_enough_books(dataset, min_books=MIN_BOOKS_PER_AUTHOR):
    books_by_author = defaultdict(set)
    for _, author, filename in dataset:
        books_by_author[author].add(filename)

    valid_authors = {
        author for author, books in books_by_author.items()
        if len(books) >= min_books
    }
    excluded_authors = sorted(set(books_by_author) - valid_authors)
    filtered = [item for item in dataset if item[1] in valid_authors]

    return filtered, sorted(valid_authors), excluded_authors, books_by_author


def choose_test_book(book_counts):
    total = sum(book_counts.values())
    target = total * 0.2
    return min(book_counts, key=lambda book: abs(book_counts[book] - target))


def author_book_split(dataset):
    books_by_author = defaultdict(lambda: defaultdict(list))

    for idx, (_, author, filename) in enumerate(dataset):
        books_by_author[author][filename].append(idx)

    train_idx = []
    test_idx = []
    split_info = {}

    for author, books in sorted(books_by_author.items()):
        if len(books) < 2:
            raise ValueError(f"{author} için book-level split yapılamaz.")

        book_counts = {book: len(indices) for book, indices in books.items()}
        test_book = choose_test_book(book_counts)
        split_info[author] = {"test_book": test_book, "books": book_counts}

        for book, indices in books.items():
            if book == test_book:
                test_idx.extend(indices)
            else:
                train_idx.extend(indices)

    return np.array(train_idx), np.array(test_idx), split_info

def author_book_split_with_selection(dataset, selected_test_books):
    books_by_author = defaultdict(lambda: defaultdict(list))

    for idx, (_, author, filename) in enumerate(dataset):
        books_by_author[author][filename].append(idx)

    train_idx = []
    test_idx = []
    split_info = {}

    for author, books in sorted(books_by_author.items()):
        test_book = selected_test_books[author]

        split_info[author] = {
            "test_book": test_book,
            "books": {book: len(indices) for book, indices in books.items()},
        }

        for book, indices in books.items():
            if book == test_book:
                test_idx.extend(indices)
            else:
                train_idx.extend(indices)

    return np.array(train_idx), np.array(test_idx), split_info

def balance_indices(labels, indices, max_per_author=None, random_state=RANDOM_STATE):
    rng = random.Random(random_state)
    by_author = defaultdict(list)

    for idx in indices:
        by_author[labels[idx]].append(idx)

    target = min(len(items) for items in by_author.values())
    if max_per_author is not None:
        target = min(target, max_per_author)

    balanced = []
    for _, items in sorted(by_author.items()):
        rng.shuffle(items)
        balanced.extend(items[:target])

    rng.shuffle(balanced)
    return np.array(balanced), target


def build_tfidf_pipeline():
    features = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 6),
            min_df=2,
            max_features=90000,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )),
    ])

    return Pipeline([
        ("features", features),
        ("clf", LinearSVC(
            C=0.75,
            class_weight="balanced",
            max_iter=6000,
            random_state=RANDOM_STATE,
        )),
    ])


def print_distribution(title, labels):
    print(f"\n{title}")
    for author, count in sorted(Counter(labels).items()):
        print(f"  {author}: {count}")


def plot_turkish_confusion_matrix(y_test, y_pred, class_names):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_ylabel("Gerçek")
    ax.set_xlabel("Tahmin")
    ax.set_title("Türkçe Yazar Tespiti - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix kaydedildi: {CONFUSION_PATH}")


def save_artifacts(pipeline, label_encoder, metadata):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(pipeline, f)
    with ENCODER_PATH.open("wb") as f:
        pickle.dump(label_encoder, f)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Production model kaydedildi: {MODEL_PATH}")
    print(f"Label encoder kaydedildi: {ENCODER_PATH}")
    print(f"Metadata kaydedildi: {METADATA_PATH}")


def train_and_validate(dataset):
    texts = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_idx, test_idx, split_info = author_book_split(dataset)
    train_idx, train_target = balance_indices(labels, train_idx)

    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\nValidation train: {len(texts_train)}")
    print(f"Validation test : {len(texts_test)}")
    print_distribution("Train sınıfları:", train_labels)
    print_distribution("Test sınıfları:", test_labels)

    print("\nTest kitapları:")
    for author, info in split_info.items():
        print(f"  {author}: {info['test_book']}")

    pipeline = build_tfidf_pipeline()
    pipeline.fit(texts_train, y_train)
    y_pred = pipeline.predict(texts_test)
    accuracy = accuracy_score(y_test, y_pred)

    class_names = label_encoder.classes_
    print("\n" + "=" * 50)
    print("Validation model: char+word TF-IDF + LinearSVC")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    ))

    plot_turkish_confusion_matrix(y_test, y_pred, class_names)

    return label_encoder, {
        "accuracy": accuracy,
        "train_size": len(texts_train),
        "test_size": len(texts_test),
        "train_distribution": dict(Counter(train_labels)),
        "test_distribution": dict(Counter(test_labels)),
        "split_info": split_info,
        "train_balance_target": train_target,
    }

def train_and_validate_with_split(dataset, selected_test_books):
    texts = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_idx, test_idx, split_info = author_book_split_with_selection(
        dataset,
        selected_test_books,
    )

    train_counts = Counter([labels[i] for i in train_idx])
    too_small_authors = {
        author: count
        for author, count in train_counts.items()
        if count < MIN_TRAIN_RECORDS_PER_AUTHOR_FOR_VALIDATION
    }

    if too_small_authors:
        return {
            "skipped": True,
            "reason": "Yetersiz train verisi",
            "train_distribution": dict(train_counts),
            "selected_test_books": selected_test_books,
        }

    train_idx, train_target = balance_indices(labels, train_idx)

    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    pipeline = build_tfidf_pipeline()
    pipeline.fit(texts_train, y_train)

    y_pred = pipeline.predict(texts_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "skipped": False,
        "accuracy": accuracy,
        "train_size": len(texts_train),
        "test_size": len(texts_test),
        "test_distribution": dict(Counter([labels[i] for i in test_idx])),
        "selected_test_books": selected_test_books,
        "split_info": split_info,
    }

def train_production_model(dataset, label_encoder):
    texts = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    all_indices = np.arange(len(dataset))
    balanced_idx, target = balance_indices(
        labels,
        all_indices,
        max_per_author=MAX_RECORDS_PER_AUTHOR,
    )

    texts_final = [texts[i] for i in balanced_idx]
    y_final = label_encoder.transform([labels[i] for i in balanced_idx])
    final_labels = [labels[i] for i in balanced_idx]

    print(f"\nProduction train: {len(texts_final)}")
    print_distribution("Production sınıfları:", final_labels)

    pipeline = build_tfidf_pipeline()
    pipeline.fit(texts_final, y_final)

    return pipeline, {
        "production_train_size": len(texts_final),
        "production_distribution": dict(Counter(final_labels)),
        "production_balance_target": target,
    }

def run_chunk_experiments():
    print("Türkçe chunk ayarı deneyleri başlıyor...")

    rows = []

    for config in EXPERIMENT_CONFIGS:
        print("\n" + "=" * 70)
        print(f"Deney: {config['name']}")
        print(f"Chunk aralığı: {config['min_words']} - {config['max_words']} kelime")

        dataset = load_sentence_records(
            min_words=config["min_words"],
            max_words=config["max_words"],
        )
        dataset, valid_authors, excluded_authors, books_by_author = (
            filter_authors_with_enough_books(dataset)
        )

        if not dataset:
            print("Bu ayar için veri üretilemedi.")
            continue

        label_encoder, validation_meta = train_and_validate(dataset)

        rows.append({
            "name": config["name"],
            "min_words": config["min_words"],
            "max_words": config["max_words"],
            "dataset_size": len(dataset),
            "accuracy": validation_meta["accuracy"],
            "train_size": validation_meta["train_size"],
            "test_size": validation_meta["test_size"],
            "train_distribution": validation_meta["train_distribution"],
            "test_distribution": validation_meta["test_distribution"],
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_DIR / "turkish_chunk_experiments.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("Chunk deney özeti:")
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True):
        print(
            f"{row['name']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"dataset={row['dataset_size']}, "
            f"train={row['train_size']}, "
            f"test={row['test_size']}"
        )

    print(f"\nSonuçlar kaydedildi: {results_path}")

def run_multi_book_validation():
    print("Çoklu book-level validation başlıyor...")

    dataset = load_sentence_records()
    dataset, valid_authors, excluded_authors, books_by_author = (
        filter_authors_with_enough_books(dataset)
    )

    book_options = {
        author: sorted(list(books))
        for author, books in books_by_author.items()
        if author in valid_authors
    }

    combinations = list(product(*[book_options[author] for author in sorted(book_options)]))
    authors = sorted(book_options)

    rows = []

    for combo in combinations:
        selected_test_books = {
            author: book
            for author, book in zip(authors, combo)
        }

        result = train_and_validate_with_split(dataset, selected_test_books)
        rows.append(result)

        combo_text = ", ".join(
            f"{author}: {book}"
            for author, book in selected_test_books.items()
        )

        if result.get("skipped"):
            print(
                f"SKIP | {result['reason']} | "
                f"Train={result['train_distribution']} | "
                f"{combo_text}"
            )
            continue

        print(
            f"Accuracy={result['accuracy']:.4f} | "
            f"Test={result['test_size']} | "
            f"{combo_text}"
        )

    valid_rows = [row for row in rows if not row.get("skipped")]
    accuracies = [row["accuracy"] for row in valid_rows]

    summary = {
        "split_count": len(valid_rows),
        "skipped_split_count": len(rows) - len(valid_rows),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "max_accuracy": float(np.max(accuracies)),
        "rows": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_DIR / "turkish_multi_book_validation.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("Çoklu book-level validation özeti:")
    print(f"Split sayısı    : {summary['split_count']}")
    print(f"Ortalama accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Std accuracy     : {summary['std_accuracy']:.4f}")
    print(f"Min accuracy     : {summary['min_accuracy']:.4f}")
    print(f"Max accuracy     : {summary['max_accuracy']:.4f}")
    print(f"Kaydedildi       : {results_path}")

def main():
    print("Türkçe kısa-paragraf yazar modeli eğitiliyor...")
    dataset = load_sentence_records()
    print(f"\nİlk dataset boyutu: {len(dataset)} chunk")

    dataset, valid_authors, excluded_authors, books_by_author = (
        filter_authors_with_enough_books(dataset)
    )

    print("\nYazar kitap sayıları:")
    for author, books in sorted(books_by_author.items()):
        status = "kullanılıyor" if author in valid_authors else "dışlandı"
        print(f"  {author}: {len(books)} kitap ({status})")

    if excluded_authors:
        print("\nDışlanan yazarlar:")
        for author in excluded_authors:
            print(f"  {author} - book-level split için yeterli kitap yok")

    print(f"\nFiltrelenmiş dataset boyutu: {len(dataset)} chunk")
    print_distribution("Filtrelenmiş dataset sınıfları:", [item[1] for item in dataset])

    label_encoder, validation_meta = train_and_validate(dataset)
    production_model, production_meta = train_production_model(dataset, label_encoder)

    metadata = {
        "model": "FeatureUnion(char_wb 4-6gram TF-IDF + word 1-3gram TF-IDF) + LinearSVC",
        "purpose": "Streamlit app production model for 90-180 word Turkish paragraphs",
        "random_state": RANDOM_STATE,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "classes": label_encoder.classes_.tolist(),
        "valid_authors": valid_authors,
        "excluded_authors": excluded_authors,
        "books_by_author": {
            author: sorted(list(books)) for author, books in books_by_author.items()
        },
        "validation": validation_meta,
        "production": production_meta,
    }
    save_artifacts(production_model, label_encoder, metadata)


if __name__ == "__main__":
    main()
