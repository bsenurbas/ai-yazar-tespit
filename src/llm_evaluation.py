import os
import pickle
import numpy as np
from collections import defaultdict, Counter
from preprocessing import clean_text, split_into_chunks
from features import build_feature_matrix, extract_features


def load_llm_texts(llm_dir="data/llm_generated"):
    """
    data/llm_generated/gpt/ ve data/llm_generated/claude/ altındaki
    metinleri yükler.
    Döndürür: [(chunk, yazar, kaynak), ...] listesi
    kaynak: 'gpt' veya 'claude'
    """
    dataset = []

    for source in os.listdir(llm_dir):
        source_path = os.path.join(llm_dir, source)

        if not os.path.isdir(source_path):
            continue

        for author in os.listdir(source_path):
            author_path = os.path.join(source_path, author)

            if not os.path.isdir(author_path):
                continue

            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"):
                    continue

                filepath = os.path.join(author_path, filename)

                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                text = clean_text(raw_text)
                chunks = split_into_chunks(text, chunk_size=150)

                for chunk in chunks:
                    dataset.append((chunk, author, source))

    return dataset


def evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names):
    """
    Her chunk için tahmin yapar.
    Döndürür: {kaynak: {yazar: {correct, total, predictions}}}
    """
    results = defaultdict(lambda: defaultdict(
        lambda: {"correct": 0, "total": 0, "predictions": []}
    ))

    for chunk, true_author, source in llm_dataset:
        feat = extract_features(chunk)
        x = np.array([[feat.get(k, 0) for k in feature_names]])
        x_scaled = scaler.transform(x)
        pred_encoded = model.predict(x_scaled)[0]
        pred_author = label_encoder.inverse_transform([pred_encoded])[0]

        is_correct = (pred_author == true_author)
        results[source][true_author]["total"] += 1
        results[source][true_author]["predictions"].append(pred_author)
        if is_correct:
            results[source][true_author]["correct"] += 1

    return results


def print_results(results):
    """GPT ve Claude sonuçlarını karşılaştırmalı yazdırır."""

    sources = sorted(results.keys())

    for source in sources:
        print(f"\n{'='*50}")
        print(f"LLM KAYNAĞI: {source.upper()}")
        print(f"{'='*50}")

        overall_correct = 0
        overall_total = 0

        for author in sorted(results[source].keys()):
            data = results[source][author]
            correct = data["correct"]
            total = data["total"]
            accuracy = correct / total if total > 0 else 0

            overall_correct += correct
            overall_total += total

            pred_counts = Counter(data["predictions"])

            print(f"\n  Yazar: {author.upper()}")
            print(f"  Doğru tespit : {correct}/{total} ({accuracy:.1%})")
            print(f"  Tahmin dağılımı: {dict(pred_counts)}")

            if accuracy >= 0.7:
                print(f"  → TANIYABİLİYOR ✓")
            else:
                print(f"  → TANIIYAMIYOR ✗")

        overall_acc = overall_correct / overall_total if overall_total > 0 else 0
        print(f"\n  GENEL DOĞRULUK: {overall_correct}/{overall_total} ({overall_acc:.1%})")

    # Karşılaştırma özeti
    print(f"\n{'='*50}")
    print("KARŞILAŞTIRMA ÖZETİ: GPT vs CLAUDE")
    print(f"{'='*50}")

    for source in sources:
        total_correct = sum(d["correct"] for d in results[source].values())
        total_all = sum(d["total"] for d in results[source].values())
        acc = total_correct / total_all if total_all > 0 else 0
        print(f"  {source.upper():10} → {total_correct}/{total_all} ({acc:.1%})")


if __name__ == "__main__":
    print("Model yükleniyor...")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("Orijinal veri yükleniyor (feature isimleri için)...")
    from preprocessing import load_author_texts
    original_dataset = load_author_texts()
    _, _, feature_names = build_feature_matrix(original_dataset)

    print("LLM metinleri yükleniyor...")
    llm_dataset = load_llm_texts()
    print(f"Toplam LLM chunk: {len(llm_dataset)}")

    print("Değerlendiriliyor...")
    results = evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names)

    print_results(results)