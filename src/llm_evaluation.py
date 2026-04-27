import os
import pickle
import numpy as np
from collections import defaultdict
from preprocessing import clean_text, split_into_chunks
from features import build_feature_matrix, extract_features


def load_llm_texts(llm_dir="data/llm_generated"):
    """
    LLM tarafından üretilen metinleri yükler.
    Döndürür: [(chunk, yazar), ...] listesi
    Her dosya birden fazla chunk'a bölünebilir.
    """
    dataset = []

    for author in os.listdir(llm_dir):
        author_path = os.path.join(llm_dir, author)

        if not os.path.isdir(author_path):
            continue

        for filename in os.listdir(author_path):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(author_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # LLM metinleri kısa olduğu için chunk_size küçük tutuyoruz
            text = clean_text(raw_text)
            chunks = split_into_chunks(text, chunk_size=150)

            for chunk in chunks:
                dataset.append((chunk, author))

    return dataset


def evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names):
    """
    LLM metinlerini modele sorar ve sonuçları analiz eder.
    
    Her metin için:
    - Tahmin edilen yazar
    - Doğru yazar
    - Doğru mu yanlış mı?
    """
    results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})

    for chunk, true_author in llm_dataset:
        # Özellik çıkar
        feat = extract_features(chunk)
        # Modelin beklediği sırada vektör oluştur
        x = np.array([[feat.get(k, 0) for k in feature_names]])
        # Ölçeklendir
        x_scaled = scaler.transform(x)
        # Tahmin
        pred_encoded = model.predict(x_scaled)[0]
        pred_author = label_encoder.inverse_transform([pred_encoded])[0]

        is_correct = (pred_author == true_author)
        results[true_author]["total"] += 1
        results[true_author]["predictions"].append(pred_author)
        if is_correct:
            results[true_author]["correct"] += 1

    return results


def print_results(results):
    """Sonuçları düzenli yazdırır."""

    print("\n" + "="*50)
    print("LLM TAKLİDİ TESPİT SONUÇLARI")
    print("="*50)

    overall_correct = 0
    overall_total = 0

    for author, data in sorted(results.items()):
        correct = data["correct"]
        total = data["total"]
        accuracy = correct / total if total > 0 else 0

        overall_correct += correct
        overall_total += total

        # Tahmin dağılımı
        from collections import Counter
        pred_counts = Counter(data["predictions"])

        print(f"\nYazar: {author.upper()}")
        print(f"  Doğru tespit : {correct}/{total} ({accuracy:.1%})")
        print(f"  Tahmin dağılımı: {dict(pred_counts)}")

        if accuracy >= 0.7:
            print(f"  → Model LLM taklidini '{author}' olarak TANIYABİLİYOR ✓")
        else:
            print(f"  → Model LLM taklidini '{author}' olarak TANIIYAMIYOR ✗")

    print(f"\n{'='*50}")
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"GENEL DOĞRULUK: {overall_correct}/{overall_total} ({overall_acc:.1%})")
    print("="*50)


if __name__ == "__main__":
    # Kayıtlı modeli yükle
    print("Model yükleniyor...")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Orijinal veriden feature_names al
    print("Orijinal veri yükleniyor (feature isimleri için)...")
    from preprocessing import load_author_texts
    original_dataset = load_author_texts()
    _, _, feature_names = build_feature_matrix(original_dataset)

    # LLM metinlerini yükle
    print("LLM metinleri yükleniyor...")
    llm_dataset = load_llm_texts()
    print(f"Toplam LLM chunk: {len(llm_dataset)}")

    # Değerlendir
    print("Değerlendiriliyor...")
    results = evaluate_llm_texts(llm_dataset, model, scaler, label_encoder, feature_names)

    # Sonuçları yazdır
    print_results(results)