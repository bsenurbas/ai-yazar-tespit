import pickle
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def build_tfidf_pipeline():
    """
    Karakter n-gramı TF-IDF + LinearSVC pipeline'ı.
    Manuel feature gerektirmez, direkt metin kullanır.
    Stilometride char n-gram çok güçlüdür.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # Kelime sınırlarına duyarlı char n-gram
            ngram_range=(3, 5),   # Tri, tetra, penta gramlar
            min_df=2,             # En az 2 belgede geçsin
            max_features=50000,   # En sık 50k n-gram
            sublinear_tf=True     # Log frekans ölçekleme
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            max_iter=2000
        ))
    ])


def train_tfidf_model(dataset, groups):
    """
    TF-IDF modelini book-level split ile eğitir.
    dataset: [(chunk, author, filename), ...]
    """
    texts = [d[0] for d in dataset]
    labels = [d[1] for d in dataset]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Book-level split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )
    train_idx, test_idx = next(splitter.split(texts, y, groups=groups))

    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("TF-IDF modeli eğitiliyor...")
    pipeline = build_tfidf_pipeline()
    pipeline.fit(texts_train, y_train)

    y_pred = pipeline.predict(texts_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print("Model: Char TF-IDF + LinearSVC")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

    # Kaydet
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("models/tfidf_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("TF-IDF modeli kaydedildi: models/tfidf_pipeline.pkl")
    return pipeline, le, acc


def evaluate_tfidf_llm(llm_dataset, pipeline, label_encoder):
    """
    TF-IDF modeliyle LLM metinlerini değerlendirir.
    """
    from collections import defaultdict, Counter

    results = defaultdict(lambda: defaultdict(
        lambda: {"correct": 0, "total": 0, "predictions": []}
    ))

    for chunk, author, source in llm_dataset:
        pred_encoded = pipeline.predict([chunk])[0]
        pred_author = str(label_encoder.classes_[pred_encoded])

        is_correct = (pred_author == author)
        results[source][author]["total"] += 1
        results[source][author]["predictions"].append(pred_author)
        if is_correct:
            results[source][author]["correct"] += 1

    print(f"\n{'='*50}")
    print("TF-IDF MODEL — LLM TAKLİDİ SONUÇLARI")
    print(f"{'='*50}")

    for source in sorted(results.keys()):
        print(f"\nKAYNAK: {source.upper()}")
        overall_correct = 0
        overall_total = 0

        for author in sorted(results[source].keys()):
            data = results[source][author]
            correct = data["correct"]
            total = data["total"]
            acc = correct / total if total > 0 else 0
            overall_correct += correct
            overall_total += total

            pred_counts = Counter(data["predictions"])
            print(f"  {author.upper()}: {correct}/{total} ({acc:.1%})")
            print(f"  Tahmin: {dict(pred_counts)}")
            print(f"  → {'TANIYABİLİYOR ✓' if acc >= 0.7 else 'TANIYAMIYOR ✗'}")

        overall_acc = overall_correct / overall_total if overall_total > 0 else 0
        print(f"\n  GENEL: {overall_correct}/{overall_total} ({overall_acc:.1%})")