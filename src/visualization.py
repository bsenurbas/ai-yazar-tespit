import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Renk ve şekil tanımları
AUTHOR_COLORS = {
    "doyle": "#2196F3",   # Mavi
    "poe":   "#F44336",   # Kırmızı
    "wells": "#4CAF50",   # Yeşil
}

SOURCE_MARKERS = {
    "original":          ("o", 80,  1.0, "Orijinal"),
    "gpt":               ("s", 60,  0.6, "GPT Normal"),
    "claude":            ("^", 60,  0.6, "Claude Normal"),
    "gpt_controlled":    ("D", 60,  0.8, "GPT Kontrollü"),
    "claude_controlled": ("P", 60,  0.8, "Claude Kontrollü"),
}


def prepare_visualization_data(original_dataset, llm_dataset, feature_names):
    """
    Orijinal ve LLM verilerini görselleştirme için hazırlar.
    Her nokta: (özellik vektörü, yazar, kaynak)
    """
    from src.features import extract_features

    all_vectors = []
    all_authors = []
    all_sources = []

    # Orijinal metinler
    for chunk, author, _ in original_dataset:
        feat = extract_features(chunk)
        vec = [feat.get(k, 0) for k in feature_names]
        all_vectors.append(vec)
        all_authors.append(author)
        all_sources.append("original")

    # LLM metinleri
    for chunk, author, source in llm_dataset:
        feat = extract_features(chunk)
        vec = [feat.get(k, 0) for k in feature_names]
        all_vectors.append(vec)
        all_authors.append(author)
        all_sources.append(source)

    X = np.array(all_vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, all_authors, all_sources


def plot_pca(X, authors, sources, output_dir="reports", sample_size=200):
    """
    PCA ile 2D görselleştirme.
    Her yazardan sample_size kadar orijinal nokta alır.
    """
    os.makedirs(output_dir, exist_ok=True)

    # PCA uygula
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Önce orijinal metinleri çiz (arka plan)
    plotted_authors = set()
    for i, (author, source) in enumerate(zip(authors, sources)):
        if source != "original":
            continue

        color = AUTHOR_COLORS.get(author, "#999999")
        marker, size, alpha, label = SOURCE_MARKERS["original"]

        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=color, marker=marker, s=size,
            alpha=alpha * 0.3,  # Orijinaller daha şeffaf
            zorder=1
        )
        plotted_authors.add(author)

    # Sonra LLM metinlerini çiz (ön plan)
    for i, (author, source) in enumerate(zip(authors, sources)):
        if source == "original":
            continue

        color = AUTHOR_COLORS.get(author, "#999999")
        marker_info = SOURCE_MARKERS.get(source, ("x", 60, 0.8, source))
        marker, size, alpha, _ = marker_info

        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=color, marker=marker, s=size,
            alpha=alpha, edgecolors="black",
            linewidths=0.5, zorder=2
        )

    # Legend — Yazarlar
    author_patches = [
        mpatches.Patch(color=color, label=author.capitalize())
        for author, color in AUTHOR_COLORS.items()
    ]

    # Legend — Kaynaklar
    source_patches = [
        plt.scatter([], [], c="gray", marker=m, s=s, label=label,
                    edgecolors="black", linewidths=0.5)
        for src, (m, s, a, label) in SOURCE_MARKERS.items()
    ]

    legend1 = ax.legend(
        handles=author_patches,
        title="Yazar", loc="upper left",
        fontsize=10, title_fontsize=11
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=source_patches,
        title="Kaynak", loc="upper right",
        fontsize=10, title_fontsize=11
    )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} varyans)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} varyans)", fontsize=12)
    ax.set_title("PCA — Stilometrik Özellik Uzayı\nOrijinal vs LLM Taklitleri",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pca_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"PCA grafiği kaydedildi: {path}")
    plt.show()


def plot_tsne(X, authors, sources, output_dir="reports"):
    """
    t-SNE ile 2D görselleştirme.
    PCA'dan daha iyi küme ayrımı gösterir.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("t-SNE hesaplanıyor (bu birkaç dakika sürebilir)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Orijinal metinler
    for i, (author, source) in enumerate(zip(authors, sources)):
        if source != "original":
            continue
        color = AUTHOR_COLORS.get(author, "#999999")
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=color, marker="o", s=80,
            alpha=0.15, zorder=1
        )

    # LLM metinleri
    for i, (author, source) in enumerate(zip(authors, sources)):
        if source == "original":
            continue
        color = AUTHOR_COLORS.get(author, "#999999")
        marker_info = SOURCE_MARKERS.get(source, ("x", 60, 0.8, source))
        marker, size, alpha, _ = marker_info
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=color, marker=marker, s=size,
            alpha=alpha, edgecolors="black",
            linewidths=0.5, zorder=2
        )

    # Legend
    author_patches = [
        mpatches.Patch(color=color, label=author.capitalize())
        for author, color in AUTHOR_COLORS.items()
    ]
    source_patches = [
        plt.scatter([], [], c="gray", marker=m, s=s, label=label,
                    edgecolors="black", linewidths=0.5)
        for src, (m, s, a, label) in SOURCE_MARKERS.items()
    ]

    legend1 = ax.legend(
        handles=author_patches,
        title="Yazar", loc="upper left",
        fontsize=10, title_fontsize=11
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=source_patches,
        title="Kaynak", loc="upper right",
        fontsize=10, title_fontsize=11
    )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE — Stilometrik Özellik Uzayı\nOrijinal vs LLM Taklitleri",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "tsne_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"t-SNE grafiği kaydedildi: {path}")
    plt.show()


if __name__ == "__main__":
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix
    from src.llm_evaluation import load_llm_texts

    print("Orijinal veri yükleniyor...")
    original_dataset = load_author_texts()

    print("LLM verisi yükleniyor...")
    llm_dataset = load_llm_texts()

    print("Özellikler çıkarılıyor...")
    chunks_and_labels = [(d[0], d[1]) for d in original_dataset]
    _, _, feature_names = build_feature_matrix(chunks_and_labels)

    print("Görselleştirme verisi hazırlanıyor...")
    X, authors, sources = prepare_visualization_data(
        original_dataset, llm_dataset, feature_names
    )

    print(f"Toplam nokta: {len(authors)}")
    print("PCA çiziliyor...")
    plot_pca(X, authors, sources)

    print("t-SNE çiziliyor...")
    plot_tsne(X, authors, sources)