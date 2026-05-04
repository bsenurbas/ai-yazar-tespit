import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import clean_text, split_into_chunks

def build_human_vs_llm_dataset(original_dataset, llm_dataset):
    """
    Orijinal ve LLM metinlerini binary sınıflandırma için hazırlar.
    Etiketler: 'human' veya 'llm'
    """
    texts = []
    labels = []
    sources = []

    # Orijinal metinler → human
    for chunk, author, _ in original_dataset:
        # Orijinal chunk'ı yeniden böl — LLM ile aynı boyut
        sub_chunks = split_into_chunks(chunk, chunk_size=300)
        sub_chunks = [c for c in sub_chunks if len(c.split()) >= 200]
        for sc in sub_chunks:
            texts.append(sc)
            labels.append("human")
            sources.append(f"original_{author}")


    # LLM metinleri → llm
    for chunk, author, source in llm_dataset:
        texts.append(chunk)
        labels.append("llm")
        sources.append(source)

    print(f"Human metinler: {labels.count('human')}")
    print(f"LLM metinler  : {labels.count('llm')}")

    return texts, labels, sources


def train_human_vs_llm(texts, labels, test_size=0.2, random_state=42):
    """
    TF-IDF + Logistic Regression ile Human vs LLM modeli eğitir.
    LR seçildi çünkü olasılık skoru verir (güven ölçümü için).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    models = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )),
            ("clf", LinearSVC(
                class_weight="balanced",
                max_iter=3000,
                random_state=42
            ))
        ]),
        "TF-IDF + Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            ))
        ]),
    }

    results = {}
    best_acc = 0
    best_name = None
    best_pipeline = None

    for name, pipeline in models.items():
        print(f"\nEğitiliyor: {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = {"pipeline": pipeline, "accuracy": acc}

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipeline = pipeline

    print(f"\nEn iyi model: {best_name} ({best_acc:.4f})")

    # En iyi modeli kaydet
    os.makedirs("models", exist_ok=True)
    with open("models/human_vs_llm.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print("Model kaydedildi: models/human_vs_llm.pkl")

    return best_pipeline, best_name, results, X_test, y_test


def plot_confusion_matrix_hvl(pipeline, X_test, y_test,
                               output_dir="reports"):
    """Human vs LLM confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["human", "llm"])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Human", "LLM"],
        yticklabels=["Human", "LLM"],
        ax=ax
    )
    ax.set_ylabel("Gerçek", fontsize=12)
    ax.set_xlabel("Tahmin", fontsize=12)
    ax.set_title("Human vs LLM — Confusion Matrix", fontsize=14,
                 fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "human_vs_llm_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix kaydedildi: {path}")
    plt.show()


def analyze_by_source(pipeline, llm_dataset, output_dir="reports"):
    """
    Her LLM kaynağı için ayrı ayrı tespit oranını gösterir.
    GPT vs Claude vs Kontrollü karşılaştırması.
    """
    from collections import defaultdict, Counter

    source_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for chunk, author, source in llm_dataset:
        pred = pipeline.predict([chunk])[0]
        source_results[source]["total"] += 1
        if pred == "llm":
            source_results[source]["correct"] += 1

    print(f"\n{'='*50}")
    print("KAYNAK BAZINDA LLM TESPİT ORANI")
    print(f"{'='*50}")

    sources = []
    rates = []

    for source, data in sorted(source_results.items()):
        rate = data["correct"] / data["total"] if data["total"] > 0 else 0
        sources.append(source)
        rates.append(rate)
        print(f"  {source:20s}: {data['correct']}/{data['total']} "
              f"({rate:.1%}) LLM olarak tespit edildi")

    # Bar chart
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = []
    for s in sources:
        if "gpt" in s and "controlled" not in s:
            colors.append("#FF9800")
        elif "claude" in s and "controlled" not in s:
            colors.append("#9C27B0")
        elif "gpt_controlled" in s:
            colors.append("#F44336")
        else:
            colors.append("#673AB7")

    bars = ax.bar(sources, rates, color=colors, edgecolor="white",
                  linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
               label="Şans seviyesi (%50)")
    ax.set_ylabel("LLM Olarak Tespit Oranı", fontsize=12)
    ax.set_title("Kaynak Bazında LLM Tespit Başarısı\n(Human vs LLM Modeli)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "human_vs_llm_by_source.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Kaynak analizi grafiği kaydedildi: {path}")
    plt.show()


if __name__ == "__main__":
    from src.preprocessing import load_author_texts
    from src.llm_evaluation import load_llm_texts

    print("Veri yükleniyor...")
    original_dataset = load_author_texts()
    llm_dataset = load_llm_texts()

    print("\nDataset hazırlanıyor...")
    texts, labels, sources = build_human_vs_llm_dataset(
        original_dataset, llm_dataset
    )

    print("\nModel eğitiliyor...")
    pipeline, best_name, results, X_test, y_test = train_human_vs_llm(
        texts, labels
    )

    plot_confusion_matrix_hvl(pipeline, X_test, y_test)
    analyze_by_source(pipeline, llm_dataset)