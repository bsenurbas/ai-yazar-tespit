import argparse
import os
from src.utils import balance_dataset


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
    dataset = balance_dataset(dataset)

    print("Özellikler çıkarılıyor...")
    X, y, feature_names = build_feature_matrix(dataset)

    print("Veri bölünüyor...")
    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X, y)
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
    dataset = balance_dataset(dataset)
    _, _, feature_names = build_feature_matrix(dataset)

    llm_dataset = load_llm_texts()
    print(f"Toplam LLM chunk: {len(llm_dataset)}")

    results = evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names)
    print_results(results)


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
        choices=["download", "train", "evaluate", "all"],
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


if __name__ == "__main__":
    main()