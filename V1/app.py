import os
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# =========================
# PATH CONFIGURATION
# =========================
# Mendapatkan path direktori V1 secara absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="MBG Smart Monitor V1", page_icon="🍱", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Background Utama */
.stApp { background-color: #FFF8F5; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4, p, div, span { font-family: "Inter", "Poppins", sans-serif; }
h1, h2, h3, h4 { color: #381E12; }

/* FIX 1: Top Bar Header Streamlit dibuat berwarna soft peach agar tegas & tidak kosong */
header[data-testid="stHeader"] {
    background-color: #FFE8DC !important;
    border-bottom: 1px solid #FFD0C0 !important;
}
header[data-testid="stHeader"] * {
    color: #5C3214 !important;
}

/* Sidebar Theme */
section[data-testid="stSidebar"] { 
    background-color: #FFF1EC; 
    border-right: 1px solid #FFE0D6; 
}
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] span { 
    color: #381E12 !important; 
    font-weight: 600; 
}

/* FIX 2: Lingkaran Radio Button (Aktif & Non-Aktif) */
section[data-testid="stSidebar"] [data-testid="stRadioButton"] [role="radiogroup"] label div[role="radio"] {
    background-color: #FFE8DC !important;
    border: 2px solid #E86A22 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadioButton"] [role="radiogroup"] label div[role="radio"][aria-checked="true"] {
    border-color: #E86A22 !important;
    background-color: #E86A22 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadioButton"] [role="radiogroup"] label div[role="radio"][aria-checked="true"] > div {
    background-color: #FFFFFF !important;
}

/* Header Component */
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
.header-left { display: flex; align-items: center; gap: 15px; }
.header-icon { width: 55px; height: 55px; border-radius: 14px; background: #FFE5DC; display: flex; align-items: center; justify-content: center; font-size: 25px; border: 1px solid #FFD4C4; }
.header-title { font-size: 32px; font-weight: 800; color: #2B1408; margin: 0; }
.header-subtitle { font-size: 14px; color: #5C3214 !important; font-weight: 500; margin-top: 3px; }
.status-badge { background: #FFEBE3; color: #C85A17; border-radius: 20px; padding: 8px 18px; font-size: 13px; font-weight: 700; border: 1px solid #FFD4C4; }

/* Cards Styling */
.dashboard-card, .custom-card, .result-card, .comment-list-card, .loading-box { 
    background: #FFFFFF; 
    border: 1px solid #FFE5DC; 
    border-radius: 18px; 
    padding: 25px; 
    box-shadow: 0 6px 20px rgba(232, 130, 98, 0.05); 
    margin-bottom: 20px; 
}
.card-title, .result-title, .upload-title, .dashboard-title { font-size: 18px; font-weight: 700; color: #2B1408; margin-bottom: 8px; }
.card-description, .result-label, .confidence-label, .dashboard-text { font-size: 13.5px; color: #5C3214 !important; line-height: 1.6; }

.model-status-badge { background: #FFF3EE; border: 1px solid #FFE0D6; border-radius: 12px; padding: 12px 16px; font-size: 13px; font-weight: 700; color: #E86A22; text-align: center; }

/* File Uploader Container & Tombol Upload */
[data-testid="stFileUploader"] { background: #FFFFFF; border: 2px dashed #FFCBD9; border-radius: 18px; padding: 12px; }
[data-testid="stFileUploader"] section { background: #FFF5F2; border-radius: 14px; }
[data-testid="stFileUploader"] button {
    background-color: #E86A22 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #C85A17 !important;
}

/* Primary Buttons */
.stButton > button { 
    background-color: #E86A22 !important; 
    color: #FFFFFF !important; 
    border-radius: 12px; 
    font-weight: 700; 
    min-height: 44px; 
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover { 
    background-color: #C85A17 !important; 
    color: #FFFFFF !important;
    border: none; 
}

/* Textarea & Placeholder */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #FFCBA4 !important;
    color: #2B1408 !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
}
.stTextArea textarea::placeholder {
    color: #8C5B3F !important;
    opacity: 1 !important;
}
.stTextArea textarea:focus {
    border-color: #E86A22 !important;
    box-shadow: 0 0 0 2px rgba(232, 106, 34, 0.2) !important;
}

/* Results & Confidence Bar */
.result-value, .sentiment-result { font-size: 30px; font-weight: 800; color: #2B1408; margin-bottom: 25px; }
.confidence-container { display: flex; align-items: center; gap: 12px; }
.confidence-bar { flex: 1; height: 12px; background: #FFEAE2; border-radius: 20px; overflow: hidden; }
.confidence-fill { height: 100%; background: #E86A22; border-radius: 20px; }
.confidence-percent { min-width: 75px; text-align: right; font-size: 16px; font-weight: 700; color: #381E12; }

/* Comment Items */
.comment-box, .comment-item { background: #FFF5F2; border: 1px solid #FFE5DC; border-radius: 13px; padding: 15px; color: #2B1408; font-size: 14px; line-height: 1.6; margin-top: 10px; }

/* Loader */
.loader { width: 42px; height: 42px; border: 4px solid #FFEAE2; border-top: 4px solid #E86A22; border-radius: 50%; animation: spin 1s linear infinite; margin: auto; }
.loading-text { margin-top: 14px; color: #5C3214; font-size: 13px; }

@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@media (max-width: 768px) { .page-header { flex-direction: column; align-items: flex-start; gap: 15px; } .header-title { font-size: 27px; } }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
@st.cache_resource
def load_fruit_model():
    class CompatibleDense(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            # Bersihkan parameter yang memicu TypeError pada Keras
            unwanted_keys = ["quantization_config", "dtype", "batch_input_shape"]
            for key in unwanted_keys:
                config.pop(key, None)
            try:
                return super().from_config(config)
            except Exception:
                # Fallback jika argumen konfigurasi masih tidak cocok
                units = config.get("units", 1)
                activation = config.get("activation", "linear")
                layer = cls(units=units, activation=activation)
                return layer

    path_inside = os.path.join(BASE_DIR, "models", "fruit_classifier", "best_scratch_cnn_apple_orange.h5")
    path_root = os.path.join(os.path.dirname(BASE_DIR), "models", "fruit_classifier", "best_scratch_cnn_apple_orange.h5")
    model_path = path_inside if os.path.exists(path_inside) else path_root

    # Gunakan safe_mode=False jika tersedia untuk melewati validasi keras baru
    try:
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            safe_mode=False, 
            custom_objects={"Dense": CompatibleDense}
        )
    except TypeError:
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={"Dense": CompatibleDense}
        )
@st.cache_resource
def load_sentiment_model():
    # Cek model di dalam V1/models atau di models/ root
    path_inside = os.path.join(BASE_DIR, "models", "sentiment_model", "classic_sentiment_model.pkl")
    path_root = os.path.join(os.path.dirname(BASE_DIR), "models", "sentiment_model", "classic_sentiment_model.pkl")
    model_path = path_inside if os.path.exists(path_inside) else path_root

    return joblib.load(model_path)

# =========================
# HELPER FUNCTIONS
# =========================
def predict_fruit(image, model):
    img_array = np.expand_dims(np.array(image.convert("RGB").resize((128, 128))).astype("float32") / 255.0, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    return ("Orange", pred) if pred >= 0.5 else ("Apple", 1 - pred)

def predict_sentiment(text, model):
    pred = model.predict([text])[0]
    return "Negative" if pred == 0 else "Neutral" if pred == 1 else "Positive"

def render_header(icon, title, subtitle, status="System Ready"):
    st.markdown(f"""
    <div class="page-header">
        <div class="header-left">
            <div class="header-icon" title="{title}">{icon}</div>
            <div>
                <div class="header-title">{title}</div>
                <div class="header-subtitle">{subtitle}</div>
            </div>
        </div>
        <div class="status-badge">{status}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# SESSION & SIDEBAR
# =========================
if "v1_comments" not in st.session_state: 
    st.session_state.v1_comments = []

st.sidebar.markdown('<div style="font-size:22px; font-weight:800; color:#E86A22; margin-bottom:25px;">🍱 MBG Smart Monitor</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Menu", ["Dashboard", "Fruit Detector", "MBG Sentiment"])

st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0")

# =========================
# PAGES
# =========================
if page == "Dashboard":
    render_header("🍱", "MBG Smart Monitor", "Monitoring dan analisis data MBG", "System Ready")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">🍎 Fruit Classification</div>
            <div class="dashboard-text">Menggunakan model CNN untuk mengklasifikasikan buah menjadi Apple atau Orange.</div>
            <div class="model-status-badge">Model: CNN &nbsp;•&nbsp; Status: Ready</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">💬 Feedback Analysis</div>
            <div class="dashboard-text">Menggunakan model Classic ML (TF-IDF + LinearSVC) untuk menganalisis sentimen feedback penerima MBG.</div>
            <div class="model-status-badge">Model: Classic ML (TF-IDF + LinearSVC) &nbsp;•&nbsp; Status: Ready</div>
        </div>""", unsafe_allow_html=True)

elif page == "Fruit Detector":
    render_header("🍎", "Fruit Detector", "Klasifikasikan buah secara otomatis", "Model: V1 | Status: Ready")
    st.markdown('<div class="upload-title">Upload Foto:</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown('<div class="card-description" style="margin-top:10px;">Foto berhasil diunggah</div>', unsafe_allow_html=True)
        st.image(image, caption="Foto yang diupload", use_container_width=True)

        if st.button("🔍 Analisis", type="primary"):
            loading = st.empty()
            loading.markdown('<div class="loading-box"><div class="loader"></div><div class="loading-text">Menganalisis gambar...</div></div>', unsafe_allow_html=True)

            fruit, confidence = predict_fruit(image, load_fruit_model())
            conf_pct = confidence * 100
            loading.empty()

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Hasil Analisis</div>
                <div class="result-label">Hasil:</div>
                <div class="result-value">{fruit}</div>
                <div class="confidence-label">Akurasi:</div>
                <div class="confidence-container">
                    <div class="confidence-bar"><div class="confidence-fill" style="width: {conf_pct}%;"></div></div>
                    <div class="confidence-percent">{conf_pct:.2f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

elif page == "MBG Sentiment":
    render_header("💬", "MBG Sentiment", "Klasifikasikan feedback masyarakat (Positif, Negatif, Netral)", "Model: V1 | Status: Ready")

    st.markdown("""
    <div class="custom-card">
        <div class="card-title">Teks Komentar</div>
        <div class="card-description">Masukkan feedback untuk mengetahui sentimen dari komentar tersebut.</div>
    </div>""", unsafe_allow_html=True)

    feedback = st.text_area("Feedback", placeholder="Ketik atau tempel komentar...", label_visibility="collapsed", height=140)

    if st.button("🔍 Analisis", type="primary"):
        if not feedback.strip():
            st.warning("Silakan masukkan feedback terlebih dahulu.")
        else:
            loading = st.empty()
            loading.markdown('<div class="loading-box"><div class="loader"></div><div class="loading-text">Menganalisis feedback...</div></div>', unsafe_allow_html=True)

            sentiment = predict_sentiment(feedback, load_sentiment_model())
            st.session_state.v1_comments.append({"text": feedback, "sentiment": sentiment})
            loading.empty()

            colors = {"Positive": "#2E7D32", "Negative": "#D32F2F", "Neutral": "#E65100"}
            s_color = colors.get(sentiment, "#E65100")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Hasil Analisis</div>
                <div class="result-label">Komentar:</div>
                <div class="comment-box">{feedback}</div>
                <div style="height:22px;"></div>
                <div class="result-label">Hasil Sentimen:</div>
                <div class="sentiment-result" style="color:{s_color};">● {sentiment}</div>
                <div class="confidence-label">Akurasi:</div>
                <div class="confidence-container">
                    <div class="confidence-bar"><div class="confidence-fill" style="width:100%;"></div></div>
                    <div class="confidence-percent">-</div>
                </div>
                <div style="font-size:12px; color:#8C5B3F; margin-top:8px;">Model V1 tidak menyediakan nilai confidence.</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.v1_comments:
        st.markdown('<div class="comment-list-card"><div class="card-title">Daftar Komentar</div><div class="card-description" style="margin-bottom:18px;">Komentar yang telah dianalisis selama sesi ini.</div>', unsafe_allow_html=True)
        colors = {"Positive": "#2E7D32", "Negative": "#D32F2F", "Neutral": "#E65100"}
        
        for item in reversed(st.session_state.v1_comments):
            s_color = colors.get(item["sentiment"], "#E65100")
            disp_text = item["text"][:80] + "..." if len(item["text"]) > 80 else item["text"]
            comment_html = f'<div class="comment-item"><span style="color:{s_color}; font-size:16px; margin-right:7px;">●</span>{disp_text}<span style="float:right; font-weight:700; color:{s_color};">{item["sentiment"]}</span></div>'
            st.markdown(comment_html, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)