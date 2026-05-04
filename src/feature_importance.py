import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def compute_feature_importance(X, y, feature_names, groups, top_n=30):
    """
    Random Forest ile feature importance hesaplar.
    Book-level split kullanır.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(splitter.split(X, y_encoded, groups=groups))

    X_train = X[train_idx]
    y_train = y_encoded[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Random Forest eğitiliyor (feature importance için)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    return top_names, top_importances, importances, indices


def plot_feature_importance(top_names, top_importances, output_dir="reports"):
    """
    Feature importance bar chart çizer.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Renk: trigram mı, manuel mi?
    colors = []
    for name in top_names:
        if name.startswith("char_"):
            colors.append("#2196F3")   # Mavi — karakter trigramı
        elif name in ["avg_word_length", "avg_sentence_length",
                      "type_token_ratio", "stopword_ratio", "word_count"]:
            colors.append("#4CAF50")   # Yeşil — leksikal
        else:
            colors.append("#FF9800")   # Turuncu — noktalama

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(
        range(len(top_names)),
        top_importances,
        color=colors,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title("En Önemli 30 Stilometrik Özellik\n(Yazar Tespitinde Katkı)",
                 fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Karakter Trigramı"),
        Patch(facecolor="#4CAF50", label="Leksikal"),
        Patch(facecolor="#FF9800", label="Noktalama"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Feature importance grafiği kaydedildi: {path}")
    plt.show()


def plot_importance_by_author(X, y, feature_names, groups,
                               output_dir="reports"):
    """
    Her yazar için en ayırt edici özellikleri gösterir.
    One-vs-Rest yaklaşımı: her yazarı diğerlerine karşı eğitir.
    """
    os.makedirs(output_dir, exist_ok=True)

    authors = sorted(set(y))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Yazar Bazında En Ayırt Edici Özellikler",
                 fontsize=14, fontweight="bold")

    colors = {"doyle": "#2196F3", "poe": "#F44336", "wells": "#4CAF50"}

    for ax, author in zip(axes, authors):
        # Binary: bu yazar vs diğerleri
        y_binary = np.array([1 if a == author else 0 for a in y])

        le = LabelEncoder()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        train_idx, _ = next(splitter.split(X, y_binary, groups=groups))

        X_train = X[train_idx]
        y_train = y_binary[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        rf = RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)

        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        top_imp = importances[top_idx]
        top_names = [feature_names[i] for i in top_idx]

        ax.barh(range(15), top_imp,
                color=colors.get(author, "#999"),
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{author.capitalize()} vs Diğerleri",
                     fontsize=12, fontweight="bold",
                     color=colors.get(author, "#999"))
        ax.set_xlabel("Importance", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance_by_author.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Yazar bazında feature importance kaydedildi: {path}")
    plt.show()


if __name__ == "__main__":
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix

    print("Veri yükleniyor...")
    dataset = load_author_texts()
    groups = [d[2] for d in dataset]
    chunks_and_labels = [(d[0], d[1]) for d in dataset]

    print("Özellikler çıkarılıyor...")
    X, y, feature_names = build_feature_matrix(chunks_and_labels)
    X = np.array(X)

    print("Feature importance hesaplanıyor...")
    top_names, top_importances, all_importances, indices = \
        compute_feature_importance(X, y, feature_names, groups, top_n=30)

    print("\nEn önemli 10 özellik:")
    for i, (name, imp) in enumerate(zip(top_names[:10], top_importances[:10])):
        print(f"  {i+1:2d}. {name:40s} {imp:.4f}")

    plot_feature_importance(top_names, top_importances)
    plot_importance_by_author(X, y, feature_names, groups)