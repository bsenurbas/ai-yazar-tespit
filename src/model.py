import os
import pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setine böler.
    test_size=0.2 → %80 eğitim, %20 test
    random_state=42 → tekrarlanabilir sonuçlar için
    """
    # Yazar isimlerini sayıya çevir (poe→0, doyle→1, wells→2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Özellikleri ölçeklendir (SVM için kritik)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded  # Her yazardan eşit oranda al
    )

    return X_train, X_test, y_train, y_test, le, scaler

def prepare_data_grouped(X, y, groups, test_size=0.2, random_state=42):
    """
    Kitap bazında train/test ayırır.
    Aynı kitabın chunk'ları hem train hem test'e düşmez.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        splitter.split(X, y_encoded, groups=groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le, scaler

def train_all_models(X_train, y_train):
    """
    3 farklı modeli eğitir ve döndürür.
    Neden 3 model? Karşılaştırma yapıp en iyisini seçeceğiz.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="linear",
            random_state=42,
            probability=True,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"Eğitiliyor: {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate_models(trained_models, X_train, X_test, y_train, y_test, label_encoder):
    """
    Her modeli değerlendirir ve sonuçları yazdırır.
    """
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)

        acc = accuracy_score(y_test, y_pred)

        # Cross-validation: modelin gerçek performansı
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"Test Accuracy : {acc:.4f}")
        print(f"CV Accuracy   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nClassification Report:")
        print(classification_report(
            y_test_labels, y_pred_labels
        ))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
        }

    return results


def plot_confusion_matrices(results, y_test, label_encoder, output_dir="reports"):
    """
    Her model için confusion matrix görselleştirir.
    Confusion matrix: hangi yazarı hangisiyle karıştırıyor?
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize=16)

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}")
        ax.set_ylabel("Gerçek")
        ax.set_xlabel("Tahmin")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    print(f"\nConfusion matrix kaydedildi: {path}")
    plt.show()


def save_best_model(results, label_encoder, scaler, output_dir="models"):
    """
    En yüksek CV accuracy'ye sahip modeli kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)

    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best_model = results[best_name]["model"]

    print(f"\nEn iyi model: {best_name} (CV: {results[best_name]['cv_mean']:.4f})")

    # Model, encoder ve scaler'ı kaydet
    with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Model kaydedildi: models/best_model.pkl")
    return best_name


if __name__ == "__main__":
    from src.preprocessing import load_author_texts
    from src.features import build_feature_matrix

    print("Veri yükleniyor...")
    dataset = load_author_texts()

    print("Özellikler çıkarılıyor...")
    X, y, feature_names = build_feature_matrix(dataset)

    print("Veri bölünüyor...")
    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X, y)

    print(f"\nEğitim seti: {X_train.shape[0]} örnek")
    print(f"Test seti  : {X_test.shape[0]} örnek")

    print("\nModeller eğitiliyor...")
    trained_models = train_all_models(X_train, y_train)

    print("\nModeller değerlendiriliyor...")
    results = evaluate_models(trained_models, X_train, X_test, y_train, y_test, le)

    plot_confusion_matrices(results, y_test, le)

    save_best_model(results, le, scaler)