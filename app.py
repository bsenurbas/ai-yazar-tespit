import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from nltk.tokenize import sent_tokenize
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text
from src.features import extract_features

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Yazar Tespit Sistemi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #8b8d98;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .result-human {
        background: #0d2137;
        border-left: 4px solid #2196F3;
        color: #90caf9;
    }
    .result-llm {
        background: #1a0d1a;
        border-left: 4px solid #e040fb;
        color: #ce93d8;
    }
    .result-author {
        background: #0d2137;
        border-left: 4px solid #00bcd4;
        color: #80deea;
    }
    .stat-box {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4a90d9;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #8b8d98;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Yazar Profil Verileri ────────────────────────────────────────────────────
AUTHOR_PROFILES_STYLOMETRY = {
    "Poe": {
        "avg_word_length": 5.2,
        "avg_sentence_length": 28.5,
        "type_token_ratio": 0.52,
        "stopword_ratio": 0.48,
        "comma_rate": 0.032,
        "semicolon_rate": 0.008,
        "question_rate": 0.004,
        "exclamation_rate": 0.003,
    },
    "Doyle": {
        "avg_word_length": 4.6,
        "avg_sentence_length": 18.2,
        "type_token_ratio": 0.58,
        "stopword_ratio": 0.52,
        "comma_rate": 0.022,
        "semicolon_rate": 0.003,
        "question_rate": 0.008,
        "exclamation_rate": 0.002,
    },
    "Wells": {
        "avg_word_length": 4.8,
        "avg_sentence_length": 22.1,
        "type_token_ratio": 0.55,
        "stopword_ratio": 0.50,
        "comma_rate": 0.025,
        "semicolon_rate": 0.004,
        "question_rate": 0.005,
        "exclamation_rate": 0.001,
    },
}

AUTHOR_PROFILES_INFO = {
    "Edgar Allan Poe": {
        "dates": "1809 — 1849",
        "style": "Gotik atmosfer, psikolojik gerilim",
        "features": [
            "Uzun, karmasik cumleler",
            "Dramatik noktalama",
            "Arkaik kelime dagarcigi",
            "Birinci sahis anlati",
        ],
        "color": "#5A07B9",
        "works": "The Raven, Tales of Mystery",
        "key": "poe",
    },
    "Arthur Conan Doyle": {
        "dates": "1859 — 1930",
        "style": "Analitik anlati, dedektif kurgu",
        "features": [
            "Diyalog agirlikli yapi",
            "Metodolojik muhakeme",
            "Victorian Ingilizce",
            "Watson anlatisi",
        ],
        "color": "#2196F3",
        "works": "Sherlock Holmes Serisi",
        "key": "doyle",
    },
    "H.G. Wells": {
        "dates": "1866 — 1946",
        "style": "Bilimsel spekulasyon, sosyal elestiri",
        "features": [
            "Gazetecilik tonu",
            "Nesnel anlati",
            "Bilim kurgu temalari",
            "Sosyal yorum",
        ],
        "color": "#4CAF50",
        "works": "The Time Machine, War of the Worlds",
        "key": "wells",
    },
}

AUTHOR_DISPLAY = {
    "poe": "Edgar Allan Poe",
    "doyle": "Arthur Conan Doyle",
    "wells": "H.G. Wells",
}

AUTHOR_COLORS = {
    "poe": "#F44336",
    "doyle": "#2196F3",
    "wells": "#4CAF50",
}

# ─── Model Yükleme ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    with open("models/tfidf_pipeline.pkl", "rb") as f:
        models["tfidf"] = pickle.load(f)
    with open("models/tfidf_label_encoder.pkl", "rb") as f:
        models["tfidf_le"] = pickle.load(f)
    with open("models/human_vs_llm.pkl", "rb") as f:
        models["hvl"] = pickle.load(f)
    return models


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
def predict_author(text, models):
    pipeline = models["tfidf"]
    le = models["tfidf_le"]
    pred = pipeline.predict([text])[0]
    author = str(le.classes_[pred])
    scores = pipeline.decision_function([text])[0]
    probs = np.exp(scores) / np.exp(scores).sum()
    return author, dict(zip([str(c) for c in le.classes_], probs))


def predict_human_vs_llm(text, models):
    pipeline = models["hvl"]
    pred = pipeline.predict([text])[0]
    try:
        proba = pipeline.predict_proba([text])[0]
        prob_dict = dict(zip(pipeline.classes_, proba))
        llm_prob = prob_dict.get("llm", 0.5)
    except Exception:
        llm_prob = 1.0 if pred == "llm" else 0.0
    return pred, llm_prob


def get_text_stats(text):
    words = text.split()
    try:
        sentences = sent_tokenize(text)
        sent_count = len(sentences)
    except Exception:
        sent_count = text.count(".") + text.count("!") + text.count("?")
    unique_words = len(set(w.lower() for w in words))
    avg_sent_len = len(words) / max(sent_count, 1)
    return {
        "Kelime Sayisi": len(words),
        "Cumle Sayisi": sent_count,
        "Benzersiz Kelime": unique_words,
        "Ort. Cumle Uzunlugu": round(avg_sent_len, 1),
    }


def get_stylometric_features(text):
    feat = extract_features(text)
    return {
        "avg_word_length": round(feat.get("avg_word_length", 0), 2),
        "avg_sentence_length": round(feat.get("avg_sentence_length", 0), 2),
        "type_token_ratio": round(feat.get("type_token_ratio", 0), 3),
        "stopword_ratio": round(feat.get("stopword_ratio", 0), 3),
        "comma_rate": round(feat.get("comma_rate", 0), 4),
        "semicolon_rate": round(feat.get("semicolon_rate", 0), 4),
        "question_rate": round(feat.get("question_rate", 0), 4),
        "exclamation_rate": round(feat.get("exclamation_rate", 0), 4),
    }


def plot_radar(features, author_key):
    labels = [
        "Kelime Uzunlugu", "Cumle Uzunlugu", "TTR",
        "Stopword", "Virgul", "Noktali Virgul"
    ]
    feat_keys = [
        "avg_word_length", "avg_sentence_length", "type_token_ratio",
        "stopword_ratio", "comma_rate", "semicolon_rate"
    ]
    max_vals = {
        "avg_word_length": 8.0, "avg_sentence_length": 50.0,
        "type_token_ratio": 1.0, "stopword_ratio": 1.0,
        "comma_rate": 0.06, "semicolon_rate": 0.02,
    }

    author_profile_key = author_key.capitalize()
    if author_profile_key not in AUTHOR_PROFILES_STYLOMETRY:
        author_profile_key = "Poe"

    ref = AUTHOR_PROFILES_STYLOMETRY[author_profile_key]

    text_vals = [min(features.get(k, 0) / max_vals[k], 1.0) for k in feat_keys]
    ref_vals = [min(ref.get(k, 0) / max_vals[k], 1.0) for k in feat_keys]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    text_vals += text_vals[:1]
    ref_vals += ref_vals[:1]

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(polar=True),
                           facecolor="#0f1117")
    ax.set_facecolor("#1e2130")

    ax.plot(angles, ref_vals, "o-", linewidth=1.5,
            color="#4a90d9", label=f"{author_profile_key} Profili", alpha=0.7)
    ax.fill(angles, ref_vals, alpha=0.15, color="#4a90d9")
    ax.plot(angles, text_vals, "o-", linewidth=2,
            color="#00e5ff", label="Analiz Edilen Metin")
    ax.fill(angles, text_vals, alpha=0.25, color="#00e5ff")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#8b8d98")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=6, color="#555")
    ax.tick_params(colors="#555")
    ax.spines["polar"].set_color("#2d3250")
    ax.grid(color="#2d3250", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=8, facecolor="#1e2130", labelcolor="#ccc",
              edgecolor="#2d3250")
    plt.tight_layout()
    return fig


def plot_feature_bars(features):
    feat_labels = {
        "avg_word_length": "Ort. Kelime Uzunlugu",
        "avg_sentence_length": "Ort. Cumle Uzunlugu",
        "type_token_ratio": "TTR",
        "stopword_ratio": "Stopword Orani",
        "comma_rate": "Virgul Orani",
        "semicolon_rate": "Noktali Virgul Orani",
        "question_rate": "Soru Isareti Orani",
        "exclamation_rate": "Unlem Orani",
    }
    keys = list(feat_labels.keys())
    labels = [feat_labels[k] for k in keys]
    values = [features.get(k, 0) for k in keys]
    max_v = max(values) if max(values) > 0 else 1
    norm_values = [v / max_v for v in values]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f1117")
    ax.set_facecolor("#1e2130")
    colors = ["#4a90d9" if v > 0.5 else "#2d5a8e" for v in norm_values]
    bars = ax.barh(labels, norm_values, color=colors,
                   edgecolor="#0f1117", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=7, color="#8b8d98")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Normalize Deger", color="#8b8d98", fontsize=8)
    ax.tick_params(colors="#8b8d98", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2d3250")
    ax.spines["left"].set_color("#2d3250")
    ax.grid(axis="x", color="#2d3250", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_gauge(prob, label):
    fig, ax = plt.subplots(figsize=(4, 2.5),
                           subplot_kw=dict(aspect="equal"),
                           facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    theta = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta), np.sin(theta),
            color="#2d3250", linewidth=15, solid_capstyle="round")
    theta_val = np.linspace(np.pi, np.pi - prob * np.pi, 100)
    color = "#e040fb" if prob > 0.5 else "#2196F3"
    ax.plot(np.cos(theta_val), np.sin(theta_val),
            color=color, linewidth=15, solid_capstyle="round")
    ax.text(0, -0.3, f"{prob:.0%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    ax.text(0, -0.6, label, ha="center", va="center",
            fontsize=9, color="#8b8d98")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.axis("off")
    plt.tight_layout()
    return fig


def render_stat_box(value, label, font_size="1.6rem"):
    return f"""
    <div class='stat-box'>
        <div class='stat-value' style='font-size:{font_size};'>{value}</div>
        <div class='stat-label'>{label}</div>
    </div>"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AI-Yazar")
    st.markdown("*Stilometri Tabanlı LLM Tespit Sistemi*")
    st.divider()

    st.markdown("### Navigasyon")
    page = st.radio(
        "",
        [
            "Yazar Tespiti",
            "Human vs LLM",
            "Stilometrik Analiz",
            "Karsilastirma Modu",
            "Stil Uretici",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### Yazar Profilleri")

    for author, info in AUTHOR_PROFILES_INFO.items():
        with st.expander(author):
            features_html = "".join([
                f"<div style='font-size:0.78rem; color:#8b8d98; "
                f"padding:0.15rem 0;'>· {f}</div>"
                for f in info["features"]
            ])
            st.markdown(f"""
            <div style='border-left: 3px solid {info["color"]};
                        padding-left: 0.8rem; padding-top: 0.3rem;'>
                <div style='font-size:0.75rem; color:#666;
                     margin-bottom:0.3rem;'>{info["dates"]}</div>
                <div style='font-size:0.82rem; color:#bbb;
                     margin-bottom:0.5rem; font-style:italic;'>
                    {info["style"]}</div>
                {features_html}
                <div style='font-size:0.72rem; color:#555;
                     margin-top:0.6rem;'>{info["works"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#555; text-align:center;'>
    YZM426 Metin Madenciligi<br>Buse Nur Baş
    </div>
    """, unsafe_allow_html=True)

# ─── Model Yükleme ────────────────────────────────────────────────────────────
try:
    models = load_models()
except Exception as e:
    st.error(f"Model yuklenemedi: {e}")
    st.stop()

# ─── SAYFA: Yazar Tespiti ─────────────────────────────────────────────────────
if page == "Yazar Tespiti":
    st.markdown("<div class='app-title'>Yazar Tespiti</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Metni girin — Poe, Doyle veya Wells'e ait olup olmadığını tespit edin.</div>",
                unsafe_allow_html=True)

    text_input = st.text_area(
        "Metin",
        height=220,
        placeholder="Analiz etmek istediginiz metni buraya yapistirin...",
        label_visibility="collapsed",
        key="author_input"
    )

    analyze_btn = st.button("Analiz Et", type="primary", key="author_btn")

    if analyze_btn and text_input:
        if len(text_input.split()) < 30:
            st.warning("Daha iyi sonuc icin en az 30 kelime girin.")
        else:
            with st.spinner("Analiz yapiliyor..."):
                clean = clean_text(text_input)
                author, probs = predict_author(clean, models)
                stats = get_text_stats(text_input)

            st.markdown(f"""
            <div class='result-box result-author'>
                Tahmin: {AUTHOR_DISPLAY.get(author, author)}
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Guven Skorlari**")
                for a, prob in sorted(probs.items(),
                                      key=lambda x: x[1], reverse=True):
                    name = AUTHOR_DISPLAY.get(a, a)
                    color = AUTHOR_COLORS.get(a, "#999")
                    st.markdown(f"""
                    <div style='display:flex; align-items:center;
                         margin-bottom:0.6rem; gap:1rem;'>
                        <div style='width:160px; color:#ccc;
                             font-size:0.9rem;'>{name}</div>
                        <div style='flex:1; background:#1e2130;
                             border-radius:4px; height:12px; overflow:hidden;'>
                            <div style='width:{prob*100:.1f}%;
                                 background:{color}; height:100%;
                                 border-radius:4px;'></div>
                        </div>
                        <div style='width:50px; text-align:right;
                             color:{color}; font-weight:600;
                             font-size:0.9rem;'>{prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Metin Istatistikleri**")
                for label, val in stats.items():
                    st.markdown(render_stat_box(val, label),
                                unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Lutfen bir metin girin.")

# ─── SAYFA: Human vs LLM ──────────────────────────────────────────────────────
elif page == "Human vs LLM":
    st.markdown("<div class='app-title'>Human vs LLM Tespiti</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Bu metin bir insan tarafından mı yoksa yapay zeka tarafından mı yazıldı?</div>",
                unsafe_allow_html=True)

    text_input = st.text_area(
        "Metin",
        height=220,
        placeholder="Analiz etmek istediginiz metni buraya yapistirin...",
        label_visibility="collapsed",
        key="hvl_input"
    )

    analyze_btn = st.button("Analiz Et", type="primary", key="hvl_btn")

    if analyze_btn and text_input:
        if len(text_input.split()) < 30:
            st.warning("Daha iyi sonuc icin en az 30 kelime girin.")
        else:
            with st.spinner("Analiz yapiliyor..."):
                clean = clean_text(text_input)
                pred, llm_prob = predict_human_vs_llm(clean, models)
                stats = get_text_stats(text_input)

            if pred == "llm":
                st.markdown("""
                <div class='result-box result-llm'>
                    Bu metin buyuk olasilikla bir YAPAY ZEKA tarafindan yazildi.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-box result-human'>
                    Bu metin buyuk olasilikla bir INSAN tarafindan yazildi.
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                fig = plot_gauge(llm_prob, "LLM Olasiligi")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                fig = plot_gauge(1 - llm_prob, "Human Olasiligi")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col3:
                st.markdown("**Metin Istatistikleri**")
                for label, val in stats.items():
                    st.markdown(render_stat_box(val, label),
                                unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Lutfen bir metin girin.")

# ─── SAYFA: Stilometrik Analiz ────────────────────────────────────────────────
elif page == "Stilometrik Analiz":
    st.markdown("<div class='app-title'>Stilometrik Analiz</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Metnin stilometrik profilini analiz edin ve yazar profilleriyle karsilastirin.</div>",
                unsafe_allow_html=True)

    text_input = st.text_area(
        "Metin",
        height=200,
        placeholder="Analiz etmek istediginiz metni buraya yapistirin...",
        label_visibility="collapsed",
        key="style_input"
    )

    analyze_btn = st.button("Analiz Et", type="primary", key="style_btn")

    if analyze_btn and text_input:
        if len(text_input.split()) < 30:
            st.warning("Daha iyi sonuc icin en az 30 kelime girin.")
        else:
            with st.spinner("Analiz yapiliyor..."):
                clean = clean_text(text_input)
                features = get_stylometric_features(clean)
                author, probs = predict_author(clean, models)
                pred, llm_prob = predict_human_vs_llm(clean, models)
                stats = get_text_stats(text_input)

            author_short = author.capitalize()

            # Özet satırı
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    render_stat_box(
                        AUTHOR_DISPLAY.get(author, author),
                        "Tahmin Edilen Yazar",
                        font_size="0.95rem"
                    ), unsafe_allow_html=True)
            with col2:
                kaynak = "LLM" if pred == "llm" else "Human"
                color = "#e040fb" if pred == "llm" else "#2196F3"
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-value' style='font-size:1.2rem;
                         color:{color};'>{kaynak}</div>
                    <div class='stat-label'>Kaynak Tahmini</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(
                    render_stat_box(stats["Kelime Sayisi"], "Kelime Sayisi"),
                    unsafe_allow_html=True)
            with col4:
                st.markdown(
                    render_stat_box(stats["Cumle Sayisi"], "Cumle Sayisi"),
                    unsafe_allow_html=True)

            st.markdown("---")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**Radar Grafigi — {author_short} Profili ile Karsilastirma**")
                fig = plot_radar(features, author)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                st.markdown("**Ozellik Dagilimi**")
                fig = plot_feature_bars(features)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.markdown("---")
            st.markdown("**Detayli Stilometrik Ozellikler**")
            feat_display = {
                "Ort. Kelime Uzunlugu": features["avg_word_length"],
                "Ort. Cumle Uzunlugu": features["avg_sentence_length"],
                "TTR": features["type_token_ratio"],
                "Stopword Orani": features["stopword_ratio"],
                "Virgul Orani": features["comma_rate"],
                "Noktali Virgul Orani": features["semicolon_rate"],
                "Soru Isareti Orani": features["question_rate"],
                "Unlem Orani": features["exclamation_rate"],
            }
            cols = st.columns(4)
            for i, (name, val) in enumerate(feat_display.items()):
                with cols[i % 4]:
                    st.markdown(
                        render_stat_box(val, name, font_size="1.2rem"),
                        unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Lutfen bir metin girin.")

# ─── SAYFA: Karşılaştırma Modu ────────────────────────────────────────────────
elif page == "Karsilastirma Modu":
    st.markdown("<div class='app-title'>Karsilastirma Modu</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Iki metni yan yana analiz edin ve stilometrik profillerini karsilastirin.</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Metin 1**")
        text1 = st.text_area(
            "",
            height=200,
            placeholder="Birinci metni buraya yapistirin...",
            key="compare_text1",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**Metin 2**")
        text2 = st.text_area(
            "",
            height=200,
            placeholder="Ikinci metni buraya yapistirin...",
            key="compare_text2",
            label_visibility="collapsed"
        )

    analyze_btn = st.button("Iki Metni Karsilastir", type="primary",
                            use_container_width=True, key="compare_btn")

    if analyze_btn and text1 and text2:
        if len(text1.split()) < 30 or len(text2.split()) < 30:
            st.warning("Her iki metin icin de en az 30 kelime girin.")
        else:
            with st.spinner("Analiz yapiliyor..."):
                clean1 = clean_text(text1)
                clean2 = clean_text(text2)
                author1, probs1 = predict_author(clean1, models)
                author2, probs2 = predict_author(clean2, models)
                pred1, llm_prob1 = predict_human_vs_llm(clean1, models)
                pred2, llm_prob2 = predict_human_vs_llm(clean2, models)
                feat1 = get_stylometric_features(clean1)
                feat2 = get_stylometric_features(clean2)
                stats1 = get_text_stats(text1)
                stats2 = get_text_stats(text2)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Metin 1")
                st.markdown(f"""
                <div class='result-box result-author'>
                    Yazar: {AUTHOR_DISPLAY.get(author1, author1)}
                </div>""", unsafe_allow_html=True)

                kaynak1 = "Yapay Zeka" if pred1 == "llm" else "Insan"
                box1 = "result-llm" if pred1 == "llm" else "result-human"
                st.markdown(f"""
                <div class='result-box {box1}'>
                    Kaynak: {kaynak1} ({llm_prob1:.0%} LLM olasiligi)
                </div>""", unsafe_allow_html=True)

                for label, val in stats1.items():
                    st.markdown(
                        render_stat_box(val, label, font_size="1.2rem"),
                        unsafe_allow_html=True)

            with col2:
                st.markdown("### Metin 2")
                st.markdown(f"""
                <div class='result-box result-author'>
                    Yazar: {AUTHOR_DISPLAY.get(author2, author2)}
                </div>""", unsafe_allow_html=True)

                kaynak2 = "Yapay Zeka" if pred2 == "llm" else "Insan"
                box2 = "result-llm" if pred2 == "llm" else "result-human"
                st.markdown(f"""
                <div class='result-box {box2}'>
                    Kaynak: {kaynak2} ({llm_prob2:.0%} LLM olasiligi)
                </div>""", unsafe_allow_html=True)

                for label, val in stats2.items():
                    st.markdown(
                        render_stat_box(val, label, font_size="1.2rem"),
                        unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Stilometrik Ozellik Karsilastirmasi")

            feat_keys = [
                "avg_word_length", "avg_sentence_length",
                "type_token_ratio", "stopword_ratio",
                "comma_rate", "semicolon_rate"
            ]
            feat_labels_map = {
                "avg_word_length": "Ort. Kelime Uzunlugu",
                "avg_sentence_length": "Ort. Cumle Uzunlugu",
                "type_token_ratio": "TTR",
                "stopword_ratio": "Stopword Orani",
                "comma_rate": "Virgul Orani",
                "semicolon_rate": "Noktali Virgul Orani",
            }

            labels = [feat_labels_map[k] for k in feat_keys]
            vals1 = [feat1.get(k, 0) for k in feat_keys]
            vals2 = [feat2.get(k, 0) for k in feat_keys]

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1117")
            ax.set_facecolor("#1e2130")
            ax.bar(x - width/2, vals1, width, label="Metin 1",
                   color="#4a90d9", edgecolor="#0f1117")
            ax.bar(x + width/2, vals2, width, label="Metin 2",
                   color="#00e5ff", edgecolor="#0f1117")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right",
                               color="#8b8d98", fontsize=8)
            ax.tick_params(colors="#8b8d98")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#2d3250")
            ax.spines["left"].set_color("#2d3250")
            ax.grid(axis="y", color="#2d3250", linewidth=0.5)
            ax.legend(facecolor="#1e2130", labelcolor="#ccc",
                      edgecolor="#2d3250", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    elif analyze_btn:
        st.warning("Lutfen her iki metin alanini da doldurun.")

# ─── SAYFA: Stil Üretici ──────────────────────────────────────────────────────
elif page == "Stil Uretici":
    st.markdown("<div class='app-title'>Stil Uretici</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Secilen yazar stilinde orijinal metin uretin ve stilometrik analizini gorun.</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Ayarlar**")

        st.info("Model: Groq — Llama 3.3 70B")
        llm_choice = "Groq (Llama 3.3)"

        author_choice = st.selectbox(
            "Yazar Stili",
            ["Edgar Allan Poe", "Arthur Conan Doyle", "H.G. Wells"]
        )

        topic_choice = st.selectbox(
            "Konu",
            [
                "Gece yarim kasabaya gelen yabanci",
                "Terk edilmis bir evin gizemi",
                "Bilimsel bir kesif",
                "Bir cinayet sorusturmasi",
                "Dogaustu bir karsilasma",
                "Ozel konu"
            ]
        )

        if topic_choice == "Ozel konu":
            topic = st.text_input("Konunuzu yazin:")
            if not topic:
                topic = "gizemli bir olay"
        else:
            topic = topic_choice

        word_count_target = st.slider(
            "Hedef Kelime Sayisi",
            min_value=100,
            max_value=500,
            value=250,
            step=50
        )

        controlled = st.checkbox(
            "Kontrollu Mod",
            value=True,
            help="Isaretlenirse karakter ve eser isimleri yasaklanir, sadece stil taklit edilir."
        )

        generate_btn = st.button("Metin Uret", type="primary",
                                 use_container_width=True)

    with col2:
        st.markdown("**Uretilen Metin**")

        if generate_btn:
            author_map = {
                "Edgar Allan Poe": "poe",
                "Arthur Conan Doyle": "doyle",
                "H.G. Wells": "wells"
            }
            author_key = author_map[author_choice]

            style_descriptions = {
                "poe": "gothic atmosphere, psychological dread, long complex sentences, rich archaic vocabulary, dramatic punctuation with semicolons and dashes, first-person unreliable narrator",
                "doyle": "analytical first-person narration, methodical reasoning stated explicitly, Victorian English, short punchy dialogue, logical deduction from physical observations",
                "wells": "matter-of-fact journalistic tone, scientific speculation, social commentary, plain direct sentences contrasting with complex ideas, early science fiction atmosphere"
            }

            forbidden_map = {
                "poe": "Do NOT mention: ravens, pendulums, Usher, Legrand, or other iconic Poe characters.",
                "doyle": "Do NOT mention: Sherlock Holmes, Watson, Baker Street, Moriarty, or any Doyle characters.",
                "wells": "Do NOT mention: time machines, Martians, invisible men, or iconic Wells story elements."
            }

            forbidden = forbidden_map[author_key] if controlled else ""

            prompt = f"""Write approximately {word_count_target} words of original prose about: {topic}

Style: Write in the style of {author_choice}.
Characteristics: {style_descriptions[author_key]}
{forbidden}

Rules:
- Pure prose only, no headings or commentary
- Start directly with the story
- Authentic to the author's period and voice"""

            with st.spinner(f"{author_choice} stilinde metin uretiliyor..."):
                try:
                    if llm_choice == "Groq (Llama 3.3)":
                        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are an expert literary stylist specializing in {author_choice}'s writing style."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.85,
                            max_tokens=800
                        )
                        generated_text = response.choices[0].message.content

                    st.text_area(
                        "",
                        value=generated_text,
                        height=280,
                        label_visibility="collapsed",
                        key="generated_output"
                    )

                    st.markdown("---")
                    st.markdown("**Stilometrik Analiz**")

                    with st.spinner("Analiz yapiliyor..."):
                        clean = clean_text(generated_text)
                        author_pred, probs = predict_author(clean, models)
                        pred_hvl, llm_prob = predict_human_vs_llm(clean, models)
                        features = get_stylometric_features(clean)
                        stats = get_text_stats(generated_text)

                    match = author_pred == author_key
                    match_color = "#4CAF50" if match else "#F44336"
                    match_text = "Basarili" if match else "Basarisiz"

                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.markdown(
                            render_stat_box(
                                AUTHOR_DISPLAY.get(author_pred, author_pred),
                                "Model Tahmini",
                                font_size="0.9rem"
                            ), unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class='stat-box'>
                            <div class='stat-value' style='font-size:1.2rem;
                                 color:{match_color};'>{match_text}</div>
                            <div class='stat-label'>Stil Taklidi</div>
                        </div>""", unsafe_allow_html=True)
                    with col_c:
                        st.markdown(
                            render_stat_box(f"{llm_prob:.0%}", "LLM Olasiligi"),
                            unsafe_allow_html=True)
                    with col_d:
                        st.markdown(
                            render_stat_box(stats["Kelime Sayisi"], "Kelime Sayisi"),
                            unsafe_allow_html=True)

                    st.markdown("**Radar Grafigi — Yazar Profili ile Karsilastirma**")
                    fig = plot_radar(features, author_key)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                except Exception as e:
                    st.error(f"Hata: {e}")
                    st.info("API key .env dosyasinda tanimli olmayabilir. GROQ_API_KEY kontrol edin.")
