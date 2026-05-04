import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix


def build_ensemble(X_manual, texts, y, groups, feature_names,
                   top_n_features=50):
    """
    TF-IDF + Manuel özellikler birleşik model.
    En önemli top_n_features manuel özelliği seçer.
    """
    from src.feature_importance import compute_feature_importance

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Book-level split
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X_manual, y_encoded, groups=groups)
    )

    # En önemli özellikleri seç
    print(f"En önemli {top_n_features} özellik seçiliyor...")
    top_names, _, all_importances, indices = compute_feature_importance(
        X_manual, y, feature_names, groups, top_n=top_n_features
    )
    top_indices = indices[:top_n_features]

    # Manuel özellikleri filtrele
    X_manual_top = X_manual[:, top_indices]

    scaler = StandardScaler()
    X_manual_scaled = scaler.fit_transform(X_manual_top)

    # TF-IDF vektörleri
    print("TF-IDF vektörleri oluşturuluyor...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(texts)

    # Train/test böl
    X_manual_train = X_manual_scaled[train_idx]
    X_manual_test = X_manual_scaled[test_idx]
    X_tfidf_train = X_tfidf[train_idx]
    X_tfidf_test = X_tfidf[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # Birleştir
    X_combined_train = hstack([
        X_tfidf_train,
        csr_matrix(X_manual_train)
    ])
    X_combined_test = hstack([
        X_tfidf_test,
        csr_matrix(X_manual_test)
    ])

    print("\nModel eğitiliyor...")
    model = LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
    model.fit(X_combined_train, y_train)

    y_pred = model.predict(X_combined_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("Model: TF-IDF + Manuel Özellikler (Ensemble)")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

    # Kaydet
    os.makedirs("models", exist_ok=True)
    with open("models/ensemble_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/ensemble_tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("models/ensemble_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/ensemble_top_indices.pkl", "wb") as f:
        pickle.dump(top_indices, f)
    with open("models/ensemble_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Ensemble model kaydedildi.")
    return model, tfidf, scaler, top_indices, le, acc


def evaluate_ensemble_llm(llm_dataset, model, tfidf, scaler,
                           top_indices, label_encoder, feature_names):
    """
    Ensemble modeliyle LLM metinlerini değerlendirir.
    """
    from collections import defaultdict, Counter
    from src.features import extract_features
    from scipy.sparse import hstack, csr_matrix

    results = defaultdict(lambda: defaultdict(
        lambda: {"correct": 0, "total": 0, "predictions": []}
    ))

    for chunk, author, source in llm_dataset:
        # Manuel özellikler
        feat = extract_features(chunk)
        x_manual = np.array([[feat.get(k, 0) for k in feature_names]])
        x_manual_top = x_manual[:, top_indices]
        x_manual_scaled = scaler.transform(x_manual_top)

        # TF-IDF
        x_tfidf = tfidf.transform([chunk])

        # Birleştir
        x_combined = hstack([x_tfidf, csr_matrix(x_manual_scaled)])

        pred_encoded = model.predict(x_combined)[0]
        pred_author = str(label_encoder.classes_[pred_encoded])

        is_correct = (pred_author == author)
        results[source][author]["total"] += 1
        results[source][author]["predictions"].append(pred_author)
        if is_correct:
            results[source][author]["correct"] += 1

    print(f"\n{'='*50}")
    print("ENSEMBLE MODEL — LLM TAKLİDİ SONUÇLARI")
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