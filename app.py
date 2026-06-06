import csv
import json
import os
import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.turkish_features import extract_turkish_features


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
TURKISH_STYLE_PROFILE_PATH = REPORTS_DIR / "turkish_author_style_profiles.json"

TURKISH_MODEL_PATH = MODELS_DIR / "turkish_tfidf_pipeline.pkl"
TURKISH_ENCODER_PATH = MODELS_DIR / "turkish_label_encoder.pkl"
FOREIGN_MODEL_PATH = MODELS_DIR / "tfidf_pipeline.pkl"
FOREIGN_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

TURKISH_AUTHOR_DISPLAY = {
    "ahmet_rasim": "Ahmet Rasim",
    "omer_seyfettin": "Ömer Seyfettin",
    "sabahattin_ali": "Sabahattin Ali",
}

FOREIGN_AUTHOR_DISPLAY = {
    "doyle": "Arthur Conan Doyle",
    "poe": "Edgar Allan Poe",
    "wells": "H. G. Wells",
}

TURKISH_AUTHOR_COLORS = {
    "ahmet_rasim": "#2f80ed",
    "omer_seyfettin": "#27ae60",
    "sabahattin_ali": "#c0392b",
}

FOREIGN_AUTHOR_COLORS = {
    "doyle": "#2f80ed",
    "poe": "#8e44ad",
    "wells": "#f39c12",
}

AUTHOR_STYLE_PROMPTS = {
    "ahmet_rasim": (
        "gözlemci, canlı, mahalle ve sokak hayatına yakın, konuşma diline yakın, "
        "hafif nükteli"
    ),
    "omer_seyfettin": (
        "sade, akıcı, olay örgüsü belirgin, kısa cümleli ve canlı"
    ),
    "sabahattin_ali": (
        "sade, içten, hüzünlü, insanın iç dünyasına ve toplumsal gerçekliğe yakın"
    ),
}

TURKISH_AUTHOR_INFO = {
    "ahmet_rasim": {
        "title": "Ahmet Rasim",
        "desc": "Şehir, mahalle ve gündelik hayat gözlemleri; konuşma diline yakın canlı anlatım.",
    },
    "omer_seyfettin": {
        "title": "Ömer Seyfettin",
        "desc": "Açık olay örgüsü, sade dil, kısa ve canlı cümlelerle ilerleyen hikaye yapısı.",
    },
    "sabahattin_ali": {
        "title": "Sabahattin Ali",
        "desc": "İçten, hüzünlü ve toplumsal gerçekliğe yakın insan odaklı anlatım.",
    },
}

FOREIGN_AUTHOR_INFO = {
    "doyle": {
        "title": "Arthur Conan Doyle",
        "desc": "Dedektif anlatısı, gözlem, olay çözümü ve mantıksal ilerleyişe yakın üslup.",
    },
    "poe": {
        "title": "Edgar Allan Poe",
        "desc": "Gotik atmosfer, psikolojik gerilim, karanlık imgeler ve yoğun iç ses.",
    },
    "wells": {
        "title": "H. G. Wells",
        "desc": "Bilimkurgu, toplumsal gözlem, fikir odaklı anlatım ve açıklayıcı ritim.",
    },
}

AUTHOR_DISPLAY = TURKISH_AUTHOR_DISPLAY
AUTHOR_COLORS = TURKISH_AUTHOR_COLORS
AUTHOR_INFO = TURKISH_AUTHOR_INFO

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "he", "her", "his", "i", "in", "is", "it",
    "its", "not", "of", "on", "or", "she", "that", "the", "their",
    "they", "this", "to", "was", "were", "with", "you",
}

OLLAMA_MODELS = {
    "Turkish-LLM-7B Q4": "hf.co/ogulcanaydogan/Turkish-LLM-7B-Instruct-GGUF:Q4_K_M",
    "Gemma3 4B": "gemma3:4b",
}

DEFAULT_TOPICS = [
    "Akşam vakti küçük bir kasabaya gelen yabancı",
    "Eski bir okul gününü hatırlayan anlatıcı",
    "Yoksul bir mahallede geçen kısa bir karşılaşma",
    "Uzun süredir beklenen bir mektubun gelişi",
    "Kış sabahı istasyonda bekleyen bir kişi",
]


st.set_page_config(
    page_title="Türkçe Üslup Madenciliği",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background: #111827;
        min-height: 82px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e5e7eb;
        line-height: 1.25;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .result-box {
        border-left: 4px solid #38bdf8;
        background: #0f172a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
    }
    .small-note {
        color: #9ca3af;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_author_model(model_path, encoder_path):
    with Path(model_path).open("rb") as f:
        pipeline = pickle.load(f)
    with Path(encoder_path).open("rb") as f:
        label_encoder = pickle.load(f)
    return pipeline, label_encoder

@st.cache_data
def load_turkish_style_profiles():
    if not TURKISH_STYLE_PROFILE_PATH.exists():
        return None

    with TURKISH_STYLE_PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def word_count(text):
    return len(text.split())


def sentence_count(text):
    marks = text.count(".") + text.count("!") + text.count("?")
    return max(marks, 1)


def predict_author(text, pipeline, label_encoder):
    pred_id = pipeline.predict([text])[0]
    if isinstance(pred_id, (int, np.integer)):
        pred_author = str(label_encoder.classes_[pred_id])
    else:
        pred_author = str(pred_id)

    scores = None
    if hasattr(pipeline, "predict_proba"):
        probs = np.atleast_1d(pipeline.predict_proba([text])[0])
        scores = {
            str(author): float(prob)
            for author, prob in zip(label_encoder.classes_, probs)
        }
    elif hasattr(pipeline, "decision_function"):
        raw = pipeline.decision_function([text])[0]
        raw = np.atleast_1d(raw)
        exp = np.exp(raw - raw.max())
        probs = exp / exp.sum()
        scores = {
            str(author): float(prob)
            for author, prob in zip(label_encoder.classes_, probs)
        }

    return pred_author, scores


def ranked_author_scores(scores):
    if not scores:
        return []
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def confidence_level(scores):
    ranked = ranked_author_scores(scores)
    if len(ranked) < 2:
        return {
            "label": "Belirsiz",
            "margin": 0,
            "top_score": 0,
            "second_score": 0,
            "second_author": None,
        }

    top_author, top_score = ranked[0]
    second_author, second_score = ranked[1]
    margin = top_score - second_score

    if margin >= 0.15:
        label = "Yüksek"
    elif margin >= 0.08:
        label = "Orta"
    elif margin >= 0.04:
        label = "Düşük"
    else:
        label = "Sınırda"

    return {
        "label": label,
        "margin": margin,
        "top_score": top_score,
        "second_score": second_score,
        "second_author": second_author,
    }

def confidence_note(scores):
    info = confidence_level(scores)

    if info["label"] == "Sınırda":
        ranked = ranked_author_scores(scores)
        top_author = ranked[0][0]
        second_author = info["second_author"]

        return (
            "Skorlar birbirine çok yakın. Bu metin kesin bir yazar etiketi yerine "
            f"{AUTHOR_DISPLAY.get(top_author, top_author)} ve "
            f"{AUTHOR_DISPLAY.get(second_author, second_author)} arasında sınırda görünüyor."
        )

    if info["label"] == "Düşük":
        second_author = info["second_author"]
        return (
            "Tahmin düşük güven düzeyinde. İkinci güçlü aday: "
            f"{AUTHOR_DISPLAY.get(second_author, second_author)}."
        )

    if info["label"] == "Orta":
        second_author = info["second_author"]
        return (
            "Tahmin orta güven düzeyinde. İkinci güçlü aday: "
            f"{AUTHOR_DISPLAY.get(second_author, second_author)}."
        )

    return None


def quality_flags(text, corpus_mode):
    flags = []
    lowered = text.lower()
    wc = word_count(text)

    if wc < 100:
        flags.append("Metin 100 kelimeden kısa; yazar tahmini güvenilir olmayabilir.")
    if any(term in lowered for term in ["görev:", "kurallar:", "açıklama", "özür", "yardımcı ol", "rules:", "sorry", "i cannot"]):
        flags.append("Metin hikaye yerine açıklama/meta cevap içeriyor olabilir.")
    if corpus_mode == "Türkçe" and any(name in text for name in ["Sarah", "Lila", "John", "Emily"]):
        flags.append("Metinde yabancı karakter adı var; Türkçe edebi profil için zayıf sinyal olabilir.")
    if "\x1b" in text or "�" in text or "anlams?z" in lowered:
        flags.append("Metinde bozuk karakter veya terminal artığı var.")

    return flags

def repetition_ratio(text):
    words = [
        word.strip(".,!?;:'\"()[]-").lower()
        for word in text.split()
    ]
    words = [word for word in words if word]

    if len(words) < 2:
        return 0

    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0

    repeated = len(bigrams) - len(set(bigrams))
    return repeated / len(bigrams)


def generation_quality_rows(text, target_author=None, predicted_author=None, corpus_mode="Türkçe"):
    flags = quality_flags(text, corpus_mode)
    wc = word_count(text)
    rep = repetition_ratio(text)

    rows = [
        {
            "Kontrol": "Kelime sayısı",
            "Durum": "Geçti" if wc >= 100 else "Uyarı",
            "Detay": f"{wc} kelime",
        },
        {
            "Kontrol": "Meta/açıklama",
            "Durum": "Uyarı" if any("meta" in flag or "açıklama" in flag for flag in flags) else "Geçti",
            "Detay": "Açıklama/meta cevap tespit edildi" if any("meta" in flag or "açıklama" in flag for flag in flags) else "Hikaye formatında",
        },
        {
            "Kontrol": "Bozuk karakter",
            "Durum": "Uyarı" if any("bozuk" in flag.lower() for flag in flags) else "Geçti",
            "Detay": "Bozuk karakter olabilir" if any("bozuk" in flag.lower() for flag in flags) else "Temiz",
        },
        {
            "Kontrol": "Tekrar oranı",
            "Durum": "Uyarı" if rep > 0.08 else "Geçti",
            "Detay": f"{rep:.3f}",
        },
    ]

    if target_author and predicted_author:
        rows.append({
            "Kontrol": "Hedef yazar eşleşmesi",
            "Durum": "Geçti" if target_author == predicted_author else "Eşleşmedi",
            "Detay": (
                f"Hedef: {AUTHOR_DISPLAY.get(target_author, target_author)} | "
                f"Tahmin: {AUTHOR_DISPLAY.get(predicted_author, predicted_author)}"
            ),
        })

    return rows

def metric_card(value, label):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def confidence_card(scores):
    info = confidence_level(scores)
    return metric_card(info["label"], "Güven Düzeyi")

def english_features(text):
    words = [word.lower() for word in text.split() if word.strip(".,!?;:'\"()[]-").isalpha()]
    clean_words = [word.strip(".,!?;:'\"()[]-").lower() for word in text.split()]
    clean_words = [word for word in clean_words if word.isalpha()]
    stopword_count = sum(1 for word in clean_words if word in ENGLISH_STOPWORDS)
    return {
        "avg_word_length": float(np.mean([len(word) for word in clean_words])) if clean_words else 0,
        "type_token_ratio": len(set(clean_words)) / len(clean_words) if clean_words else 0,
        "stopword_ratio": stopword_count / len(clean_words) if clean_words else 0,
        "comma_rate": text.count(",") / max(len(text), 1),
        "semicolon_rate": text.count(";") / max(len(text), 1),
        "word_count": len(words),
    }


def text_stats(text, corpus_mode):
    features = extract_turkish_features(text) if corpus_mode == "Türkçe" else english_features(text)
    wc = word_count(text)
    sc = sentence_count(text)
    stats = {
        "Kelime": wc,
        "Cümle": sc,
        "Ort. Cümle": round(wc / max(sc, 1), 1),
        "Ort. Kelime": round(features.get("avg_word_length", 0), 2),
        "TTR": round(features.get("type_token_ratio", 0), 3),
        "Stopword": round(features.get("stopword_ratio", 0), 3),
        "Virgül": round(features.get("comma_rate", 0), 4),
    }
    if corpus_mode == "Türkçe":
        stats["Fiil Eki"] = round(features.get("verb_suffix_ratio", 0), 3)
    else:
        stats["Noktalı Virgül"] = round(features.get("semicolon_rate", 0), 4)
    return stats

STYLE_FEATURE_LABELS = {
    "avg_word_length": "Ort. kelime uzunluğu",
    "avg_sentence_length": "Ort. cümle uzunluğu",
    "type_token_ratio": "TTR",
    "stopword_ratio": "Stopword oranı",
    "verb_suffix_ratio": "Fiil eki oranı",
    "comma_rate": "Virgül oranı",
}


def closest_author_by_feature(value, profiles, feature_name):
    distances = []

    for author, profile in profiles["profiles"].items():
        feature_stats = profile["features"].get(feature_name)
        if not feature_stats:
            continue

        mean = feature_stats["mean"]
        std = feature_stats["std"] or 1

        z_distance = abs(value - mean) / std
        distances.append((z_distance, author, mean))

    if not distances:
        return None

    distances.sort(key=lambda item: item[0])
    return distances[0]


def style_similarity_rows(text):
    profiles = load_turkish_style_profiles()
    if not profiles:
        return []

    features = extract_turkish_features(text)
    rows = []

    for feature_name, label in STYLE_FEATURE_LABELS.items():
        value = features.get(feature_name, 0)
        closest = closest_author_by_feature(value, profiles, feature_name)

        if not closest:
            continue

        distance, author, author_mean = closest
        rows.append({
            "Özellik": label,
            "Metin Değeri": round(value, 4),
            "En Yakın Yazar": AUTHOR_DISPLAY.get(author, author),
            "Yazar Ort.": round(author_mean, 4),
            "Uzaklık": round(distance, 3),
        })

    return rows

def plot_scores(scores):
    labels = [AUTHOR_DISPLAY.get(k, k) for k in scores]
    values = [scores[k] for k in scores]
    colors = [AUTHOR_COLORS.get(k, "#64748b") for k in scores]

    fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0b1120")
    ax.set_facecolor("#111827")
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.tick_params(colors="#d1d5db")
    ax.set_xlabel("Normalize karar skoru", color="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.grid(axis="x", color="#374151", alpha=0.35)
    plt.tight_layout()
    return fig


def plot_features(stats):
    keys = [key for key in ["Ort. Cümle", "Ort. Kelime", "TTR", "Stopword", "Fiil Eki", "Noktalı Virgül", "Virgül"] if key in stats]
    values = [float(stats[k]) for k in keys]
    max_value = max(values) if max(values) else 1
    norm = [v / max_value for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0b1120")
    ax.set_facecolor("#111827")
    ax.bar(keys, norm, color="#38bdf8")
    ax.tick_params(colors="#d1d5db", labelrotation=20)
    ax.set_ylabel("Normalize değer", color="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.grid(axis="y", color="#374151", alpha=0.35)
    plt.tight_layout()
    return fig


def find_ollama_executable():
    candidates = [
        "ollama",
        str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            continue
    return None


def build_generation_prompt(author_key, topic, min_words, max_words):
    style = AUTHOR_STYLE_PROMPTS[author_key]
    return f"""Aşağıdaki görevi eksiksiz yerine getir.

{min_words} ile {max_words} kelime arasında özgün bir Türkçe edebi hikaye yaz.

Konu: {topic}.

Üslup: {style}.

Kurallar:
- Kısa cevap verme.
- Metin en az {min_words} kelime olmalı.
- Açıklama, başlık, özür veya uyarı yazma.
- Maddeleme yapma.
- Yabancı karakter adı kullanma.
- Sadece hikayeyi yaz.
- Bozuk veya anlamsız kelime kullanma.
"""


def generate_with_ollama(model_name, prompt):
    ollama = find_ollama_executable()
    if not ollama:
        raise RuntimeError("Ollama bulunamadı. Ollama kurulu ve PATH erişilebilir olmalı.")

    result = subprocess.run(
        [ollama, "run", model_name, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=480,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Ollama üretimi başarısız oldu.")
    return result.stdout.strip()


def read_experiment_results(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


with st.sidebar:
    st.title("Üslup Madenciliği")
    st.caption("Yazar tahmini, yerel üretim ve stilometrik analiz")
    st.divider()

    corpus_mode = st.radio(
        "Yazar Grubu",
        ["Türkçe", "Yabancı"],
        horizontal=True,
    )

    page = st.radio(
        "Sayfa",
        [
            "Yazar Tahmini",
            "Stilometrik Analiz",
            "Yerel Üretim",
            "Deney Sonuçları",
        ],
    )

    st.divider()
    st.subheader("Yazar Profilleri")

    if corpus_mode == "Türkçe":
        AUTHOR_DISPLAY = TURKISH_AUTHOR_DISPLAY
        AUTHOR_COLORS = TURKISH_AUTHOR_COLORS
        AUTHOR_INFO = TURKISH_AUTHOR_INFO
        model_path = TURKISH_MODEL_PATH
        encoder_path = TURKISH_ENCODER_PATH
    else:
        AUTHOR_DISPLAY = FOREIGN_AUTHOR_DISPLAY
        AUTHOR_COLORS = FOREIGN_AUTHOR_COLORS
        AUTHOR_INFO = FOREIGN_AUTHOR_INFO
        model_path = FOREIGN_MODEL_PATH
        encoder_path = FOREIGN_ENCODER_PATH

    for key, info in AUTHOR_INFO.items():
        with st.expander(info["title"]):
            st.write(info["desc"])


try:
    pipeline, label_encoder = load_author_model(str(model_path), str(encoder_path))
except Exception as exc:
    st.error(f"{corpus_mode} model yüklenemedi: {exc}")
    st.stop()


if page == "Yazar Tahmini":
    st.title("Yazar Tahmini")
    st.write(f"Bir {corpus_mode.lower()} metin girin; sistem metnin hangi yazara daha yakın olduğunu tahmin eder.")

    placeholder = "Analiz edilecek Türkçe metni buraya yapıştırın..." if corpus_mode == "Türkçe" else "Paste the English text to analyze here..."
    text = st.text_area("Metin", height=260, placeholder=placeholder)
    if st.button("Tahmin Et", type="primary"):
        if not text.strip():
            st.warning("Lütfen metin girin.")
        else:
            pred, scores = predict_author(text, pipeline, label_encoder)
            flags = quality_flags(text, corpus_mode)
            stats = text_stats(text, corpus_mode)

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="metric-label">Tahmin Edilen Yazar</div>
                    <div class="metric-value">{AUTHOR_DISPLAY.get(pred, pred)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if flags:
                for flag in flags:
                    st.warning(flag)

            note = confidence_note(scores)
            if note:
                st.info(note)

            display_stats = {"Güven": confidence_level(scores)["label"]}
            display_stats.update(stats)

            cols = st.columns(4)
            for idx, (label, value) in enumerate(display_stats.items()):
                with cols[idx % 4]:
                    st.markdown(metric_card(value, label), unsafe_allow_html=True)

            if scores:
                st.subheader("Karar Skorları")
                fig = plot_scores(scores)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            
            if corpus_mode == "Türkçe":
                rows = style_similarity_rows(text)
                if rows:
                    st.subheader("Stil Yakınlığı")
                    st.dataframe(rows, use_container_width=True, hide_index=True)


elif page == "Stilometrik Analiz":
    st.title("Stilometrik Analiz")
    st.write("Metnin temel stilometrik özelliklerini çıkarır ve görselleştirir.")

    placeholder = "Analiz edilecek Türkçe metni buraya yapıştırın..." if corpus_mode == "Türkçe" else "Paste the English text to analyze here..."
    text = st.text_area("Metin", height=260, placeholder=placeholder)
    if st.button("Analiz Et", type="primary"):
        if not text.strip():
            st.warning("Lütfen metin girin.")
        else:
            stats = text_stats(text, corpus_mode)
            flags = quality_flags(text, corpus_mode)

            if flags:
                for flag in flags:
                    st.warning(flag)

            cols = st.columns(4)
            for idx, (label, value) in enumerate(stats.items()):
                with cols[idx % 4]:
                    st.markdown(metric_card(value, label), unsafe_allow_html=True)

            fig = plot_features(stats)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if corpus_mode == "Türkçe":
                rows = style_similarity_rows(text)
                if rows:
                    st.subheader("Stil Yakınlığı")
                    st.dataframe(rows, use_container_width=True, hide_index=True)


elif page == "Yerel Üretim":
    st.title("Yerel LLM ile Üretim")
    st.write("Ollama üzerinden yerel modelle metin üretir, sonra çıkan metni aynı sınıflandırıcıyla değerlendirir.")

    if corpus_mode != "Türkçe":
        st.info("Yerel üretim deneyleri şu an Türkçe yazarlar için yapılandırıldı. Yabancı modda tahmin ve stilometrik analiz sayfalarını kullanabilirsin.")
        st.stop()

    generated = ""

    control_cols = st.columns(3)
    with control_cols[0]:
        model_label = st.selectbox("Yerel Model", list(OLLAMA_MODELS.keys()))
    with control_cols[1]:
        author_key = st.selectbox(
            "Hedef Üslup",
            list(AUTHOR_DISPLAY.keys()),
            format_func=lambda key: AUTHOR_DISPLAY[key],
        )
    with control_cols[2]:
        topic = st.selectbox("Konu", DEFAULT_TOPICS)

    control_cols = st.columns([1.4, 1, 1, 0.8])
    with control_cols[0]:
        custom_topic = st.text_input("Özel konu (opsiyonel)")
        if custom_topic.strip():
            topic = custom_topic.strip()
    with control_cols[1]:
        min_words = st.slider("Minimum kelime", 80, 180, 120, 10)
    with control_cols[2]:
        max_words = st.slider("Maksimum kelime", 120, 260, 170, 10)
    with control_cols[3]:
        st.write("")
        st.write("")
        run_generation = st.button("Yerel Metin Üret", type="primary")

    if run_generation:
        model_name = OLLAMA_MODELS[model_label]
        prompt = build_generation_prompt(author_key, topic, min_words, max_words)
        with st.spinner("Yerel model çalışıyor; bu işlem biraz sürebilir..."):
            try:
                generated = generate_with_ollama(model_name, prompt)
            except Exception as exc:
                st.error(str(exc))
                generated = ""

    if generated:
        st.subheader("Üretilen Metin")
        st.text_area("Üretilen Metin", value=generated, height=260, label_visibility="collapsed")

    if generated:
        st.divider()

        pred, scores = predict_author(generated, pipeline, label_encoder)
        stats = text_stats(generated, corpus_mode)
        flags = quality_flags(generated, corpus_mode)
        target_match = pred == author_key

        st.markdown(
            f"""
            <div class="result-box">
                <div class="metric-label">Üretim Değerlendirmesi</div>
                <div class="metric-value">{AUTHOR_DISPLAY.get(pred, pred)}</div>
                <div class="small-note">Hedef üslup: {AUTHOR_DISPLAY[author_key]} · {'Eşleşti' if target_match else 'Eşleşmedi'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if flags:
            for flag in flags:
                st.warning(flag)

        note = confidence_note(scores)
        if note:
            st.info(note)

        st.subheader("Üretim Kalite Kontrolü")
        quality_rows = generation_quality_rows(
            generated,
            target_author=author_key,
            predicted_author=pred,
            corpus_mode=corpus_mode,
        )
        st.dataframe(quality_rows, use_container_width=True, hide_index=True)

        display_stats = {"Güven": confidence_level(scores)["label"]}
        display_stats.update(stats)

        cols = st.columns(4)
        for idx, (label, value) in enumerate(display_stats.items()):
            with cols[idx % 4]:
                st.markdown(metric_card(value, label), unsafe_allow_html=True)

        if scores:
            st.subheader("Karar Skorları")
            fig = plot_scores(scores)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if corpus_mode == "Türkçe":
            rows = style_similarity_rows(generated)
            if rows:
                st.subheader("Stil Yakınlığı")
                st.dataframe(rows, use_container_width=True, hide_index=True)


elif page == "Deney Sonuçları":
    st.title("Deney Sonuçları")
    st.write("Projede yapılan yerel üretim ve sınıflandırma deneylerinin özetleri.")

    if corpus_mode == "Türkçe":
        st.subheader("Türkçe Yazar Sınıflandırıcı")

        classifier_rows = [
            {
                "Değerlendirme": "Production book-level split",
                "Sonuç": "%100.00",
                "Detay": "100-200 kelimelik chunk; test kitapları: Gecelerim, Bomba, Ses",
            },
            {
                "Değerlendirme": "Çoklu book-level validation",
                "Sonuç": "Ort. %90.49",
                "Detay": "20 geçerli split; std %9.34; min %73.68; max %100",
            },
            {
                "Değerlendirme": "Production eğitim seti",
                "Sonuç": "222 örnek",
                "Detay": "Yazar başına 74 dengeli örnek",
            },
            {
                "Değerlendirme": "Manuel gerçek eser testi",
                "Sonuç": "3/3 doğru",
                "Detay": "Üç yazarın kendi eserlerinden alınan pasajlar doğru sınıflandırıldı",
            },
        ]

        st.table(classifier_rows)

        st.markdown(
            """
            Kullanılan nihai model TF-IDF karakter n-gram + kelime n-gram özellikleri ile LinearSVC sınıflandırıcısından oluşur.
            Hüseyin Rahmi, yalnızca tek kitap bulunduğu için kitap bazlı doğrulama yapısına dahil edilmemiştir.
            """
        )

        st.subheader("Yerel LLM Üretim Deneyleri")
        summary_rows = [
            {"Deney": "Gemma3 4B Prompt-Only", "Sonuç": "3/12", "Başarı": "%25.0", "Not": "Ömer Seyfettin sınıfına kayma görüldü."},
            {"Deney": "Qwen2.5 0.5B LoRA", "Sonuç": "-", "Başarı": "Başarısız", "Not": "Teknik eğitim tamamlandı, üretim kalitesi bozuk çıktı."},
            {"Deney": "Qwen2.5 1.5B LoRA", "Sonuç": "-", "Başarı": "Başarısız", "Not": "Instruct ve base denemelerinde anlam bütünlüğü korunamadı."},
            {"Deney": "Turkish-LLM-7B Q4", "Sonuç": "5/15", "Başarı": "%33.3", "Not": "Okunabilirlik arttı; hedef üslup eşleşmesi sınırlı kaldı, kalite kontrol gerekli."},        ]
        st.table(summary_rows)

        results_path = REPORTS_DIR / "turkish_llm_7b_q4_long_results.csv"
        rows = read_experiment_results(results_path)
        if rows:
            st.subheader("Turkish-LLM-7B Q4 Detaylı Sonuç")
            st.dataframe(rows, use_container_width=True)
    else:
        st.subheader("Yabancı Yazar Sınıflandırıcı")
        st.markdown(
            """
            - Model: TF-IDF + LinearSVC
            - Kullanılan yazarlar: Arthur Conan Doyle, Edgar Allan Poe, H. G. Wells
            - Amaç: İngilizce metinlerde klasik yazar üslubuna yakınlık tahmini
            - Not: Bu mod yerel LLM üretiminden bağımsız, sınıflandırma demosu olarak çalışır.
            """
        )

    st.info(
        "Sınıflandırıcı sonucu, metnin eğitim korpusundaki yazarlara stilistik yakınlığını ölçer; "
        "tek başına edebi kalite veya gerçek yazarlık kanıtı değildir. "
        "Bu nedenle uygulamada karar skorları, güven düzeyi, stilometrik metrikler ve kalite uyarıları birlikte gösterilir."
    )
