import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import torch dan transformers lebih dulu
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import library lain setelahnya
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf



# =========================================================
# PAGE CONFIG & PATHS
# =========================================================
st.set_page_config(page_title="MBG Smart Monitor V2", page_icon="🍱", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit_classifier", "fixed_mobilenetv2_apple_orange.h5")
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "feedback_history.csv")

# =========================================================
# HELPER & CUSTOM CSS (WARDAH / SOFT BLUE THEME)
# =========================================================
# Palette Guide:
# - Primary Accent: #3A92A6 (Wardah Deep Cyan / Ocean Teal)
# - Soft Accent / Fill: #5CB6C9 & #8FD3DE
# - Light Backgrounds: #F2F8F9 & #E6F3F5
# - Dark High-Contrast Text: #0F2937 (Headers) & #204051 (Body)
# - Subdued Text: #4A6E7F
# - Borders: #BCE2E8 & #D3EDF2

def render_html(content):
    st.markdown(content.strip(), unsafe_allow_html=True)

render_html("""
<style>
    .stApp { 
        background-color: #F4F9FA; 
    }
    .main .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
        max-width: 1200px; 
    }
    h1, h2, h3, h4, p, div, span, label { 
        font-family: "Inter", "Poppins", sans-serif; 
        color: #102C3D; 
    }
    h1, h2, h3, h4 { 
        color: #0C2331 !important; 
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { 
        background-color: #FFFFFF; 
        border-right: 1px solid #D6EBF0; 
    }
    section[data-testid="stSidebar"] * { 
        color: #102C3D; 
    }

    /* ONLY: Navigation selected text */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="button"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="button"] * {
        color: #FFFFFF !important;
    }

    /* ONLY: Navigation dropdown option text */
    body [data-baseweb="menu"] [role="option"],
    body [data-baseweb="menu"] [role="option"] * {
        color: #FFFFFF !important;
    }

    /* ONLY: Feedback textarea placeholder text */
    textarea::placeholder {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* ONLY: Feedback textarea typed text */
    textarea {
        color: #FFFFFF !important;
    }
    
    /* Headers & Status Badges */
    .page-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 30px; 
    }
    .header-left { 
        display: flex; 
        align-items: center; 
        gap: 15px; 
    }
    .header-icon { 
        width: 50px; 
        height: 50px; 
        border-radius: 14px; 
        background: #E0F4F7; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 26px; 
        border: 1px solid #B8E4EC;
    }
    .header-title { 
        font-size: 32px; 
        font-weight: 750; 
        color: #0D2636; 
        margin: 0; 
    }
    .header-subtitle { 
        font-size: 14px; 
        color: #4A6E7F; 
        margin-top: 3px; 
        font-weight: 500;
    }
    .status-badge { 
        background: #E2F3F6; 
        color: #1B5868; 
        border-radius: 20px; 
        padding: 8px 16px; 
        font-size: 13px; 
        font-weight: 700; 
        border: 1px solid #B4E1EA; 
    }
    
    /* Cards */
    .custom-card { 
        background: #FFFFFF; 
        border: 1px solid #CFE9EE; 
        border-radius: 18px; 
        padding: 24px; 
        box-shadow: 0 6px 22px rgba(35, 95, 115, 0.05); 
        margin-bottom: 20px; 
    }
    .card-title { 
        font-size: 18px; 
        font-weight: 750; 
        color: #0E2838; 
        margin-bottom: 6px; 
    }
    .card-description { 
        font-size: 13px; 
        color: #486E80; 
        margin-bottom: 18px; 
        line-height: 1.5;
    }
    
    /* File Uploader Wardah Outline */
    .upload-title { 
        font-size: 15px; 
        font-weight: 700; 
        color: #102C3D; 
        margin-bottom: 8px; 
    }
    [data-testid="stFileUploader"] { 
        background: #FFFFFF; 
        border: 2px dashed #7ECDD9 !important; 
        border-radius: 18px; 
        padding: 12px; 
    }
    [data-testid="stFileUploader"] section { 
        background: #F2FAFB !important; 
        border-radius: 14px; 
    }
    
    /* Buttons */
    .stButton > button { 
        background: linear-gradient(135deg, #3A92A6 0%, #2A798C 100%) !important; 
        color: #FFFFFF !important; 
        border-radius: 12px !important; 
        font-weight: 700 !important; 
        min-height: 46px !important; 
        border: none !important; 
        box-shadow: 0 4px 14px rgba(42, 121, 140, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover { 
        background: linear-gradient(135deg, #2D7B8E 0%, #1E6070 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(42, 121, 140, 0.35) !important;
        transform: translateY(-1px);
    }
    
    /* Results & Previews */
    .result-card { 
        background: #FFFFFF; 
        border: 1px solid #CFE9EE; 
        border-radius: 20px; 
        padding: 25px; 
        box-shadow: 0 8px 25px rgba(35, 95, 115, 0.06); 
        margin-top: 20px; 
    }
    .fruit-preview { 
        height: 250px; 
        background: #F2F9FA; 
        border-radius: 18px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 105px; 
        border: 1px solid #DBF0F3; 
    }
    .result-label { 
        font-size: 13px; 
        font-weight: 600; 
        color: #55798C; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 4px; 
    }
    .result-value { 
        font-size: 32px; 
        font-weight: 800; 
        color: #0E2B3D; 
        margin-bottom: 22px; 
    }
    .confidence-label { 
        font-size: 13px; 
        font-weight: 600; 
        color: #385E70; 
        margin-bottom: 8px; 
    }
    .confidence-container { 
        display: flex; 
        align-items: center; 
        gap: 12px; 
    }
    .confidence-bar { 
        flex: 1; 
        height: 13px; 
        background: #E2F2F5; 
        border-radius: 20px; 
        overflow: hidden; 
    }
    .confidence-fill { 
        height: 100%; 
        background: linear-gradient(90deg, #4FB6C9, #2C7E92); 
        border-radius: 20px; 
    }
    .confidence-percent { 
        font-size: 16px; 
        font-weight: 750; 
        color: #103447; 
        min-width: 65px; 
        text-align: right; 
    }
    .fruit-indicator { 
        font-size: 20px; 
    }
    
    /* Wardah Info Tags */
    .menu-box { 
        background: #ECF7F9; 
        border: 1px solid #B8E4EC; 
        border-radius: 12px; 
        padding: 12px 18px; 
        color: #1E5C6B; 
        font-size: 13px; 
        font-weight: 600; 
        margin-top: 10px; 
    }
    .sentiment-header { 
        display: flex; 
        align-items: center; 
        gap: 14px; 
        margin-bottom: 6px; 
    }
    .sentiment-logo { 
        width: 48px; 
        height: 48px; 
        background: #256B7D; 
        border-radius: 14px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        color: #FFFFFF; 
        font-size: 24px; 
        box-shadow: 0 4px 12px rgba(37, 107, 125, 0.2); 
    }
    .sentiment-title { 
        font-size: 32px; 
        font-weight: 750; 
        color: #0C2331; 
    }
    .sentiment-description { 
        color: #466C7E; 
        font-size: 14px; 
        margin-bottom: 8px; 
        font-weight: 500;
    }
    .model-version { 
        display: inline-block; 
        background: #E1F2F5; 
        color: #215D6D; 
        padding: 5px 12px; 
        border-radius: 12px; 
        font-size: 12px; 
        font-weight: 700; 
        border: 1px solid #BCE3EB; 
        margin-bottom: 24px; 
    }
    .sentiment-emoji { 
        height: 200px; 
        background: #F3F9FA; 
        border: 1px solid #CFE9EE; 
        border-radius: 18px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 90px; 
    }
    .sentiment-result { 
        font-size: 26px; 
        font-weight: 800; 
        color: #0E2B3D; 
        margin-bottom: 20px; 
    }
    .segmented-bar { 
        display: flex; 
        gap: 6px; 
        width: 100%; 
        margin-top: 10px; 
    }
    .segment { 
        height: 12px; 
        flex: 1; 
        border-radius: 20px; 
        background: #DCECF0; 
    }
    .segment.active { 
        background: linear-gradient(90deg, #53BDD1, #2C7E92); 
    }
    .sentiment-percentage { 
        font-size: 26px; 
        font-weight: 800; 
        color: #12374B; 
        margin-top: 12px; 
    }
    .comment-box { 
        background: #F4FAFB; 
        border: 1px solid #C9E8EE; 
        border-radius: 14px; 
        padding: 16px; 
        color: #163B4E; 
        font-size: 14.5px; 
        line-height: 1.6; 
        font-weight: 500;
    }
    
    /* Metrics Summary Cards */
    .summary-card { 
        background: #FFFFFF; 
        border: 1px solid #CFE9EE; 
        border-radius: 18px; 
        padding: 22px; 
        box-shadow: 0 5px 20px rgba(35, 95, 115, 0.04); 
    }
    .summary-number { 
        font-size: 30px; 
        font-weight: 800; 
        color: #143E50; 
    }
    .summary-label { 
        color: #4C7385; 
        font-size: 13px; 
        font-weight: 600; 
        margin-top: 2px;
    }
    
    /* Loader Wardah */
    .loading-box { 
        height: 250px; 
        background: #F4F9FA; 
        border-radius: 18px; 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        border: 1px solid #D6EBF0; 
    }
    .loader { 
        width: 44px; 
        height: 44px; 
        border: 4px solid #CEEFF4; 
        border-top: 4px solid #2B7A8D; 
        border-radius: 50%; 
        animation: spin 1s linear infinite; 
    }
    @keyframes spin { 
        0% { transform: rotate(0deg); } 
        100% { transform: rotate(360deg); } 
    }
    .loading-text { 
        margin-top: 14px; 
        color: #386072; 
        font-size: 13.5px; 
        font-weight: 600; 
    }
    @media (max-width: 768px) { 
        .page-header { flex-direction: column; align-items: flex-start; gap: 15px; } 
        .header-title, .sentiment-title { font-size: 26px; } 
    }
</style>
""")

# =========================================================
# MODELS & PREPROCESSING
# =========================================================
@st.cache_resource
def load_fruit_model():
    class CompatibleDense(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config.pop("quantization_config", None)
            return super().from_config(config)
    return tf.keras.models.load_model(FRUIT_MODEL_PATH, compile=False, custom_objects={"Dense": CompatibleDense})

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
    model.eval()
    return tokenizer, model

def preprocess_image(image):
    image = image.convert("RGB").resize((128, 128))
    image_array = np.expand_dims(np.array(image), axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

def predict_fruit(image):
    model = load_fruit_model()
    prediction = model.predict(preprocess_image(image), verbose=0)
    confidence = float(prediction[0][0])
    return ("Orange", confidence) if confidence >= 0.5 else ("Apple", 1 - confidence)

def predict_sentiment(text):
    tokenizer, model = load_sentiment_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels.get(prediction, str(prediction)), confidence

def save_feedback(feedback, sentiment, confidence):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    new_data = pd.DataFrame({"feedback": [feedback], "sentiment": [sentiment], "confidence": [confidence]})
    if os.path.exists(HISTORY_PATH):
        updated_data = pd.concat([pd.read_csv(HISTORY_PATH), new_data], ignore_index=True)
    else:
        updated_data = new_data
    updated_data.to_csv(HISTORY_PATH, index=False)

# =========================================================
# SIDEBAR
# =========================================================
render_html('<div style="font-size:22px; font-weight:750; color:#12374B; margin-bottom:25px;">🍱 MBG Smart Monitor</div>')
page = st.sidebar.selectbox("Navigation", ["Dashboard", "Fruit Scan", "Feedback Analysis"])
st.sidebar.markdown("---")
st.sidebar.caption("Version 2.0")

# =========================================================
# MAIN PAGES
# =========================================================
if page == "Dashboard":
    render_html("""
<div class="page-header">
    <div class="header-left">
        <div class="header-icon">🍱</div>
        <div>
            <div class="header-title">MBG Smart Monitor</div>
            <div class="header-subtitle">Monitoring dan analisis data MBG secara terpusat</div>
        </div>
    </div>
    <div class="status-badge">System Ready</div>
</div>
    """)
    st.write("Selamat datang, Admin!")
    
    col1, col2 = st.columns(2)
    with col1:
        render_html("""
<div class="custom-card">
    <div class="card-title">🍎 Fruit Classification</div>
    <div class="card-description">Menggunakan model MobileNetV2 untuk mengklasifikasikan buah menjadi Apple atau Orange.</div>
    <div class="menu-box">Model: MobileNetV2 &nbsp; • &nbsp; Status: Ready</div>
</div>
        """)
    with col2:
        render_html("""
<div class="custom-card">
    <div class="card-title">💬 Feedback Analysis</div>
    <div class="card-description">Menggunakan model Transformer untuk mendeteksi sentimen kepuasan penerima program MBG.</div>
    <div class="menu-box">Model: Transformer &nbsp; • &nbsp; Status: Ready</div>
</div>
        """)

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        render_html('<div class="card-title" style="margin-top:15px; margin-bottom:12px;">Feedback Summary</div>')
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="summary-card"><div class="summary-number">{len(history)}</div><div class="summary-label">Total Feedback</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="summary-card"><div class="summary-number" style="color:#25879B;">{(history["sentiment"] == "Positive").sum()}</div><div class="summary-label">Positive Feedback</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="summary-card"><div class="summary-number" style="color:#BC5A65;">{(history["sentiment"] == "Negative").sum()}</div><div class="summary-label">Negative Feedback</div></div>', unsafe_allow_html=True)
    else:
        st.info("Belum ada feedback yang tersimpan.")

elif page == "Fruit Scan":
    render_html("""
<div class="page-header">
    <div>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <div class="header-title">Fruit Detector</div>
            <span style="color:#4A7182; font-size:14px; font-weight:600;">Klasifikasikan jenis buah secara otomatis</span>
        </div>
    </div>
    <div class="status-badge">Model: V2 | Status: Ready</div>
</div>
<div class="upload-title">Upload Foto:</div>
    """)
    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        render_html('<div class="card-description" style="color:#205C6B; font-weight:600;">✓ Foto berhasil diunggah</div>')
        st.image(image, caption="Foto yang diunggah", use_container_width=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            analyze_button = st.button("🔍 Analisis", type="primary", use_container_width=True)
        with col2:
            menu_placeholder = st.empty()
            menu_placeholder.markdown("""
<div style="display:flex; align-items:center; gap:8px; padding-top:8px; flex-wrap:wrap;">
    <span style="color:#1C4456; font-size:14px; font-weight:700;">Menu MBG hari ini:</span>
    <span style="background:#E2F3F6; color:#185360; border:1px solid #B0DFE8; border-radius:10px; padding:7px 14px; font-size:12.5px; font-weight:700;">Menunggu hasil analisis...</span>
</div>
            """, unsafe_allow_html=True)

        if analyze_button:
            loading_ph = st.empty()
            loading_ph.markdown('<div class="result-card"><div class="loading-box"><div class="loader"></div><div class="loading-text">Menganalisis gambar buah...</div></div></div>', unsafe_allow_html=True)
            result, confidence = predict_fruit(image)
            loading_ph.empty()
            
            fruit_emoji = "🍎" if result == "Apple" else "🍊"
            result_id = "Apel" if result == "Apple" else "Jeruk"
            
            menu_placeholder.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; padding-top:8px; flex-wrap:wrap;">
    <span style="color:#1C4456; font-size:14px; font-weight:700;">Menu MBG hari ini:</span>
    <span style="background:#DCF3F7; color:#104754; border:1px solid #99DDE9; border-radius:10px; padding:7px 14px; font-size:13px; font-weight:800;">{result_id} {fruit_emoji}</span>
</div>
            """, unsafe_allow_html=True)

            render_html(f"""
<div class="result-card">
    <div style="font-size:18px; font-weight:750; color:#0E2B3D; margin-bottom:20px;">Hasil Analisis</div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:25px;">
        <div class="fruit-preview">{fruit_emoji}</div>
        <div style="display:flex; flex-direction:column; justify-content:center;">
            <div class="result-label">Hasil Prediksi:</div>
            <div class="result-value">{result}</div>
            <div class="confidence-label">Tingkat Keyakinan:</div>
            <div class="confidence-container">
                <div class="confidence-bar"><div class="confidence-fill" style="width: {confidence*100}%;"></div></div>
                <div class="confidence-percent">{confidence*100:.2f}%</div>
                <div class="fruit-indicator">{fruit_emoji}</div>
            </div>
        </div>
    </div>
</div>
            """)

elif page == "Feedback Analysis":
    render_html("""
<div class="sentiment-header">
    <div class="sentiment-logo">✦</div>
    <div class="sentiment-title">MBG Sentiment</div>
</div>
<div class="sentiment-description">Klasifikasikan umpan balik penerima manfaat program MBG secara otomatis</div>
<div class="model-version">Model Transformer V02</div>
<div class="custom-card">
    <div class="card-title">Teks Feedback / Komentar</div>
    <div class="card-description">Ketik atau tempel opini publik atau siswa untuk dievaluasi tingkat kepuasannya.</div>
</div>
    """)
    feedback = st.text_area("Teks Komentar", placeholder="Contoh: Menu makanannya sangat bergizi dan buahnya segar...", label_visibility="collapsed", height=140)
    if st.button("🔍 Analisis Sentimen", type="primary"):
        if not feedback.strip():
            st.warning("Silakan masukkan teks komentar terlebih dahulu.")
        else:
            loading_ph = st.empty()
            loading_ph.markdown('<div class="custom-card"><div class="sentiment-emoji"><div><div class="loader" style="margin:auto;"></div><div class="loading-text">Memproses sentimen bahasa...</div></div></div></div>', unsafe_allow_html=True)
            sentiment, confidence = predict_sentiment(feedback)
            save_feedback(feedback, sentiment, confidence)
            loading_ph.empty()

            emoji = "😊" if sentiment == "Positive" else "😠" if sentiment == "Negative" else "😐"
            render_html(f"""
<div class="custom-card">
    <div class="card-title">Komentar Teranalisis</div>
    <div class="comment-box">{feedback}</div>
    <div style="height:20px;"></div>
    <div style="display:grid; grid-template-columns: 1fr 2fr; gap:20px;">
        <div class="sentiment-emoji">{emoji}</div>
        <div>
            <div class="result-label">Sentimen Terdeteksi:</div>
            <div class="sentiment-result">{sentiment}</div>
            <div class="confidence-label">Tingkat Keyakinan:</div>
            <div class="segmented-bar">
                <div class="segment active"></div><div class="segment active"></div><div class="segment active"></div><div class="segment active"></div>
            </div>
            <div class="sentiment-percentage">{confidence*100:.2f}%</div>
        </div>
    </div>
</div>
            """)

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        pos = (history["sentiment"] == "Positive").sum()
        neg = (history["sentiment"] == "Negative").sum()
        net = (history["sentiment"] == "Neutral").sum()
        total = len(history)
        
        p_pct = (pos / total) * 100 if total else 0
        n_pct = (neg / total) * 100 if total else 0
        
        # Wardah Soft Blue Donut Palette:
        # Positive: #3A99AC (Wardah Deep Cyan)
        # Negative: #D96874 (Soft Muted Coral/Red)
        # Neutral: #8DB5C4 (Soft Slate Blue)
        gradient = f"conic-gradient(#3A99AC 0% {p_pct}%, #D96874 {p_pct}% {p_pct + n_pct}%, #8DB5C4 {p_pct + n_pct}% 100%)"

        col1, col2 = st.columns([2, 1])
        with col1:
            render_html(f"""
<div class="custom-card">
    <div class="card-title">Distribusi Sentimen</div>
    <div class="card-description">Rangkuman proporsi respon dari seluruh data feedback yang tersimpan.</div>
    <div style="display:flex; justify-content:center; align-items:center; padding:20px;">
        <div style="width:220px; height:220px; border-radius:50%; background:{gradient}; box-shadow: inset 0 0 0 26px #FFFFFF, 0 8px 24px rgba(35, 95, 115, 0.08);"></div>
    </div>
    <div style="display:flex; justify-content:space-around; margin-top:15px;">
        <div style="text-align:center;"><div style="font-size:24px; font-weight:800; color:#2C8899;">{pos}</div><div class="summary-label">Positif</div></div>
        <div style="text-align:center;"><div style="font-size:24px; font-weight:800; color:#C4525E;">{neg}</div><div class="summary-label">Negatif</div></div>
        <div style="text-align:center;"><div style="font-size:24px; font-weight:800; color:#50788A;">{net}</div><div class="summary-label">Netral</div></div>
    </div>
</div>
            """)
        with col2:
            render_html('<div class="custom-card"><div class="card-title">Histori Feedback</div><div class="card-description">5 Masukan paling baru.</div></div>')
            for _, row in history.tail(5).iloc[::-1].iterrows():
                short = str(row["feedback"])[:35] + "..." if len(str(row["feedback"])) > 35 else str(row["feedback"])
                emj = "😊" if row["sentiment"] == "Positive" else "😠" if row["sentiment"] == "Negative" else "😐"
                render_html(f'<div style="background:#FFFFFF; border:1px solid #CFE9EE; border-radius:12px; padding:12px; margin-bottom:8px; box-shadow:0 2px 8px rgba(40,110,130,0.03);"><div style="font-size:13.5px; color:#1C4052; font-weight:500;">{emj} {short}</div></div>')
            
            st.download_button(
                "📄 Unduh Riwayat Lengkap (CSV)", 
                data=history.to_csv(index=False).encode("utf-8"), 
                file_name="feedback_history.csv", 
                mime="text/csv", 
                use_container_width=True
            )
    else:
        st.info("Belum ada data feedback untuk ditampilkan.")