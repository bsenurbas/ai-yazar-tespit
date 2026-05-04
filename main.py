import argparse
import os
from src.utils import balance_dataset
from src.model import prepare_data_grouped

def run_download():
    print("\n=== VERİ İNDİRME ===")
    from src.data_collector import download_books
    download_books()


def run_train():
    print("\n=== MODEL EĞİTİMİ ===")
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix
    from src.model import prepare_data, train_all_models, evaluate_models, \
        plot_confusion_matrices, save_best_model

    print("Veri yükleniyor...")
    dataset = load_author_texts()

    # Book-level split için kitap isimlerini ayır
    chunks_and_labels = [(d[0], d[1]) for d in dataset]
    groups = [d[2] for d in dataset]

    print("Özellikler çıkarılıyor...")
    X, y, feature_names = build_feature_matrix(chunks_and_labels)

    print("Veri bölünüyor (book-level)...")
    X_train, X_test, y_train, y_test, le, scaler = prepare_data_grouped(
        X, y, groups
    )
    print(f"Eğitim: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("Modeller eğitiliyor...")
    trained_models = train_all_models(X_train, y_train)

    print("Değerlendiriliyor...")
    results = evaluate_models(trained_models, X_train, X_test, y_train, y_test, le)

    plot_confusion_matrices(results, y_test, le)
    save_best_model(results, le, scaler)


def run_evaluate():
    print("\n=== LLM TAKLİDİ DEĞERLENDİRME ===")
    import pickle
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix
    from src.llm_evaluation import load_llm_texts, evaluate_llm_texts, print_results

    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    dataset = load_author_texts()
    chunks_and_labels = [(d[0], d[1]) for d in dataset]
    _, _, feature_names = build_feature_matrix(chunks_and_labels)

    llm_dataset = load_llm_texts()
    print(f"Toplam LLM chunk: {len(llm_dataset)}")

    results = evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names)
    print_results(results)

def run_tfidf():
    print("\n=== TF-IDF MODEL ===")
    from src.preprocessing import load_author_texts
    from src.tfidf_model import train_tfidf_model, evaluate_tfidf_llm
    from src.llm_evaluation import load_llm_texts

    dataset = load_author_texts()
    groups = [d[2] for d in dataset]

    pipeline, le, acc = train_tfidf_model(dataset, groups)

    print("\nLLM metinleri değerlendiriliyor...")
    llm_dataset = load_llm_texts()
    evaluate_tfidf_llm(llm_dataset, pipeline, le)

def run_visualize():
    print("\n=== GÖRSELLEŞTİRME ===")
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix
    from src.llm_evaluation import load_llm_texts
    from src.visualization import prepare_visualization_data, plot_pca, plot_tsne

    print("Orijinal veri yükleniyor...")
    original_dataset = load_author_texts()

    print("LLM verisi yükleniyor...")
    llm_dataset = load_llm_texts()

    print("Özellik isimleri çıkarılıyor...")
    chunks_and_labels = [(d[0], d[1]) for d in original_dataset]
    _, _, feature_names = build_feature_matrix(chunks_and_labels)

    print("Görselleştirme verisi hazırlanıyor...")
    X, authors, sources = prepare_visualization_data(
        original_dataset, llm_dataset, feature_names
    )

    print(f"Toplam nokta: {len(authors)}")
    plot_pca(X, authors, sources)
    plot_tsne(X, authors, sources)

def run_all():
    run_download()
    run_train()
    run_evaluate()


def main():
    parser = argparse.ArgumentParser(
        description="AI Yazar Tespiti — Stilometri Tabanlı",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["download", "train", "evaluate","tfidf","visualize", "all"],
        required=True,
        help=(
            "download  → Kitapları indir\n"
            "train     → Modeli eğit\n"
            "evaluate  → LLM tespiti yap\n"
            "all       → Hepsini sırayla çalıştır"
        )
    )
    args = parser.parse_args()

    if args.mode == "download":
        run_download()
    elif args.mode == "train":
        run_train()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "all":
        run_all()
    elif args.mode == "tfidf":
        run_tfidf()
    elif args.mode == "visualize":
        run_visualize()


if __name__ == "__main__":
    main()