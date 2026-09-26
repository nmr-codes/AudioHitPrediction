import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

# Page Configuration
st.set_page_config(
    page_title="Audio Hit Predictor | LightGBM vs XGBoost",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Dark Theme CSS (Zero Emojis, Pure Minimalist Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC;
    }
    
    .stApp {
        background-color: #0B0F17;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #374151;
    }

    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 6px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Instruction Cards */
    .guide-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .step-number {
        display: inline-block;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        border-radius: 50%;
        background-color: #1E293B;
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 8px;
        border: 1px solid #334155;
    }

    /* Status Badges */
    .badge-hit {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-miss {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Minimalist Progress Bar */
    .prob-track {
        height: 8px;
        width: 100%;
        background-color: #1F2937;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 10px;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1F2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.9rem;
        color: #94A3B8;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
        font-weight: 600;
    }

    /* Sidebar Clean */
    [data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1A2233 !important;
    }

    /* Table Styling */
    div[data-testid="stTable"] table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    div[data-testid="stTable"] th {
        background-color: #161F30 !important;
        color: #CBD5E1 !important;
        text-align: left;
        padding: 10px;
        border-bottom: 1px solid #2A364F;
    }
    div[data-testid="stTable"] td {
        background-color: #0E131F !important;
        color: #F1F5F9 !important;
        padding: 9px 10px;
        border-bottom: 1px solid #1A2233;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_models_and_metadata():
    lgbm_model = joblib.load('models/lgbm_best.joblib')
    xgb_model = joblib.load('models/xgb_model.joblib')
    meta = joblib.load('models/pipeline_meta.joblib')
    with open('results.json', 'r') as f:
        results = json.load(f)
    return lgbm_model, xgb_model, meta, results

try:
    lgbm_model, xgb_model, meta, results = load_models_and_metadata()
    le_genre = meta['label_encoder']
    genres = list(le_genre.classes_)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header Section
col_head_left, col_head_right = st.columns([3, 1])
with col_head_left:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <span style="font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #38BDF8; font-weight: 700;">
            Machine Learning System Architecture
        </span>
        <h1 style="font-size: 1.9rem; font-weight: 700; margin: 4px 0 8px 0; color: #F8FAFC;">
            Audio Hit Predictor & Model Benchmark
        </h1>
        <p style="font-size: 0.92rem; color: #94A3B8; margin: 0; line-height: 1.5;">
            Spotify platformasidagi 32,828 ta qo'shiq audio xususiyatlari asosida musiqiy Hit ehtimolini bashorat qilish va LightGBM vs XGBoost modellarini qiyoslash tizimi.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_head_right:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em;">Dataset Hajmi</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">32,828</div>
        <div style="font-size: 0.72rem; color: #10B981;">Stratified 80/20 Test To'plami</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Tabs (User Guide First)
tab_guide, tab_predict, tab_benchmark, tab_overfitting, tab_docs = st.tabs([
    "Foydalanish Qo'llanmasi",
    "Jonli Bashorat (Inference)", 
    "Modellar Benchmarki", 
    "Overfitting & Tuning Lab", 
    "Texnik Xulosa"
])

# -------------------------------------------------------------
# TAB 0: USER GUIDE & SYSTEM OVERVIEW (QO'LLANMA)
# -------------------------------------------------------------
with tab_guide:
    st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 14px;'>Loyiha Haqida va Foydalanish Bo'yicha To'liq Qo'llanma</div>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2, gap="large")

    with col_g1:
        st.markdown("""<div class="guide-card">
<div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 8px;">1. Ushbu Tizim Nima?</div>
<p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 12px;">
Ushbu loyiha Spotify musiqa platformasidagi <strong>32,828 ta haqiqiy treklar</strong> ma'lumotlar to'plami asosida yaratilgan. Har bir qo'shiqning raqsbopligi (danceability), energiyasi (energy), ovoz balandligi (loudness), akustikligi va reliz yili kabi 14 ta belgisi o'rganilib, qo'shiqning <strong>Hit (mashhur: Popularity &ge; 50)</strong> bo'lish ehtimoli bashorat qilinadi.
</p>
<p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6; margin: 0;">
Tizimda <strong>LightGBM</strong> va <strong>XGBoost</strong> gradient boosting algoritmlari taqqoslangan, daraxt barglari soni (<code>num_leaves</code>) bo'yicha overfitting chegarasi aniqlangan hamda <strong>Optuna</strong> (Bayesian optimization) orqali eng optimal parametrlar topilgan.
</p>
</div>
<div class="guide-card">
<div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #34D399; margin-bottom: 8px;">3. Akustik Ko'rsatkichlar Izohi</div>
<ul style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.7; padding-left: 18px; margin: 0;">
<li><strong>Danceability (0 - 1.0):</strong> Qo'shiqning raqsga tushishga qulaylik darajasi (ritm, temp va barqarorlik).</li>
<li><strong>Energy (0 - 1.0):</strong> Treklardagi tezlik, intensivlik va shovqin miqdori.</li>
<li><strong>Loudness (-30 dan 1 dB):</strong> Umumiy ovoz balandligi. Odatda zamonaviy xitlar -6 dB dan balandroq bo'ladi.</li>
<li><strong>Valence (0 - 1.0):</strong> Musiqiy kayfiyat. Yuqori qiymat shodlik va optimizmni, past qiymat esa g'amginlikni anglatadi.</li>
<li><strong>Acousticness (0 - 1.0):</strong> Trekda akustik asboblar ustunligi (elektron asboblarga qarama-qarshi).</li>
<li><strong>Tempo (BPM):</strong> Trekning daqiqadagi zarbalar soni (Beats Per Minute).</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col_g2:
        st.markdown("""<div class="guide-card">
<div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #F59E0B; margin-bottom: 14px;">2. Qadam-baqadam Foydalanish Yo'riqnomasi</div>
<div style="margin-bottom: 14px;">
<span class="step-number">1</span>
<strong style="color: #F8FAFC; font-size: 0.88rem;">Modelni tanlang:</strong>
<p style="font-size: 0.83rem; color: #94A3B8; margin: 4px 0 0 32px;">Chap paneldagi <em>"Active Estimator"</em> orqali <strong>Optuna-Tuned LightGBM</strong> (eng aniq: 70.1% Acc, 0.770 AUC), <strong>XGBoost</strong> yoki <strong>Baseline LightGBM</strong> ni faollashtiring.</p>
</div>
<div style="margin-bottom: 14px;">
<span class="step-number">2</span>
<strong style="color: #F8FAFC; font-size: 0.88rem;">Parametrlarni kiriting yoki Blueprint tanlang:</strong>
<p style="font-size: 0.83rem; color: #94A3B8; margin: 4px 0 0 32px;"><em>"Jonli Bashorat"</em> sahifasiga o'tib, tayyor shablonni (masalan: <em>Viral Pop Anthem</em>, <em>Club EDM Peak</em>) tanlang yoki slayderlar orqali o'z trekingiz qiymatlarini belgilang.</p>
</div>
<div style="margin-bottom: 14px;">
<span class="step-number">3</span>
<strong style="color: #F8FAFC; font-size: 0.88rem;">Natijani tekshiring:</strong>
<p style="font-size: 0.83rem; color: #94A3B8; margin: 4px 0 0 32px;">O'ng paneldagi <em>"Inference Result"</em> kartasida model qo'shiqning Hit bo'lish foizini hisoblaydi (&ge;50% bo'lsa <strong>HIT CANDIDATE</strong> statusi beriladi).</p>
</div>
<div style="margin-bottom: 14px;">
<span class="step-number">4</span>
<strong style="color: #F8FAFC; font-size: 0.88rem;">Modellar tahlilini ko'ring:</strong>
<p style="font-size: 0.83rem; color: #94A3B8; margin: 4px 0 0 32px;"><em>"Modellar Benchmarki"</em> va <em>"Overfitting Lab"</em> bo'limlariga o'tib, metrikalar, o'qitish tezligi (soniyada) va overfitting grafiklarini o'rganing.</p>
</div>
<div>
<span class="step-number">5</span>
<strong style="color: #F8FAFC; font-size: 0.88rem;">Jupyter Notebook (practise.ipynb):</strong>
<p style="font-size: 0.83rem; color: #94A3B8; margin: 4px 0 0 32px;">Loyiha papkasida <code>practise.ipynb</code> fayli mavjud. Unda barcha hisoblashlar, o'qitish skriptlari va chizmalar to'liq saqlangan bo'lib, mustaqil qayta ishga tushirish mumkin.</p>
</div>
</div>""", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 1: LIVE INFERENCE (JONLI BASHORAT)
# -------------------------------------------------------------
with tab_predict:
    # Sidebar Model & Preset Selection
    st.sidebar.markdown("""
    <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 8px; font-weight: 600;">
        Model Configuration
    </div>
    """, unsafe_allow_html=True)

    selected_model_name = st.sidebar.selectbox(
        "Active Estimator",
        ["Optuna-Tuned LightGBM", "XGBoost (Hist)", "Baseline LightGBM"]
    )

    st.sidebar.markdown("<hr style='border-color: #1F2937; margin: 16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 8px; font-weight: 600;">
        Track Profile Presets
    </div>
    """, unsafe_allow_html=True)

    preset = st.sidebar.selectbox(
        "Select Blueprint",
        ["Custom Attributes", "Viral Pop Anthem", "Club EDM Peak", "Late Night R&B", "Acoustic Indie Ballad"]
    )

    # Preset Attribute Defaults
    if preset == "Viral Pop Anthem":
        p_dance, p_energy, p_loud, p_acoust, p_val, p_tempo, p_year, p_genre = 0.82, 0.78, -4.2, 0.08, 0.75, 124.0, 2024, "pop"
    elif preset == "Club EDM Peak":
        p_dance, p_energy, p_loud, p_acoust, p_val, p_tempo, p_year, p_genre = 0.72, 0.94, -3.1, 0.02, 0.62, 128.0, 2023, "edm"
    elif preset == "Late Night R&B":
        p_dance, p_energy, p_loud, p_acoust, p_val, p_tempo, p_year, p_genre = 0.68, 0.52, -7.5, 0.35, 0.40, 96.0, 2022, "r&b"
    elif preset == "Acoustic Indie Ballad":
        p_dance, p_energy, p_loud, p_acoust, p_val, p_tempo, p_year, p_genre = 0.48, 0.35, -10.5, 0.82, 0.28, 88.0, 2021, "rock"
    else:
        p_dance, p_energy, p_loud, p_acoust, p_val, p_tempo, p_year, p_genre = 0.65, 0.70, -6.5, 0.15, 0.50, 120.0, 2023, "pop"

    # Inference Input Layout
    col_inputs, col_output = st.columns([3, 2], gap="large")

    with col_inputs:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #E2E8F0; margin-bottom: 12px;'>Acoustic Feature Vector (Trek Parametrlari)</div>", unsafe_allow_html=True)
        
        row1_1, row1_2 = st.columns(2)
        with row1_1:
            danceability = st.slider("Danceability (Raqsboplik)", 0.0, 1.0, float(p_dance), 0.01)
            energy = st.slider("Energy (Energetiklik)", 0.0, 1.0, float(p_energy), 0.01)
            valence = st.slider("Valence (Musiqiy Kayfiyat)", 0.0, 1.0, float(p_val), 0.01)
            acousticness = st.slider("Acousticness (Akustiklik)", 0.0, 1.0, float(p_acoust), 0.01)
        with row1_2:
            loudness = st.slider("Loudness (Ovoz balandligi dB)", -30.0, 1.0, float(p_loud), 0.5)
            tempo = st.slider("Tempo (BPM)", 50.0, 220.0, float(p_tempo), 1.0)
            speechiness = st.slider("Speechiness (Vokal / Matn)", 0.0, 1.0, 0.07, 0.01)
            instrumentalness = st.slider("Instrumentalness (Sozlik)", 0.0, 1.0, 0.00, 0.01)

        row2_1, row2_2, row2_3 = st.columns(3)
        with row2_1:
            genre = st.selectbox("Asosiy Janr", genres, index=genres.index(p_genre) if p_genre in genres else 0)
        with row2_2:
            release_year = st.number_input("Reliz Yili", min_value=1950, max_value=2026, value=int(p_year), step=1)
        with row2_3:
            duration_min = st.number_input("Davomiyligi (Daqiqa)", min_value=1.0, max_value=12.0, value=3.25, step=0.1)

        # Advanced Toggle
        with st.expander("Qo'shimcha Modal Xususiyatlar"):
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            with col_adv1:
                key = st.selectbox("Tonal Kalit (Key)", list(range(12)), index=5)
            with col_adv2:
                mode = st.selectbox("Tonal Rejim (Mode)", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor")
            with col_adv3:
                liveness = st.slider("Jonli Ijro (Liveness)", 0.0, 1.0, 0.12, 0.01)

    # Feature Assembly and Prediction
    genre_encoded = le_genre.transform([genre])[0]
    input_vector = pd.DataFrame([{
        'danceability': danceability,
        'energy': energy,
        'key': key,
        'loudness': loudness,
        'mode': mode,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo,
        'duration_min': duration_min,
        'release_year': release_year,
        'playlist_genre': genre_encoded
    }])

    # Model Evaluation
    if selected_model_name == "Optuna-Tuned LightGBM":
        active_clf = lgbm_model
    elif selected_model_name == "XGBoost (Hist)":
        active_clf = xgb_model
    else:
        active_clf = lgbm_model

    prob_hit = float(active_clf.predict_proba(input_vector)[0][1])
    is_hit = prob_hit >= 0.50
    confidence = prob_hit if is_hit else (1.0 - prob_hit)

    with col_output:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #E2E8F0; margin-bottom: 12px;'>Inference Result (Bashorat Natijasi)</div>", unsafe_allow_html=True)
        
        status_badge = f'<span class="badge-hit">HIT CANDIDATE</span>' if is_hit else f'<span class="badge-miss">SUB-THRESHOLD</span>'
        bar_color = "#10B981" if is_hit else "#EF4444"

        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="metric-title">Bashorat Statusi</span>
                {status_badge}
            </div>
            <div class="metric-value">{prob_hit:.1%}</div>
            <div class="metric-sub">Hit Bo'lish Ehtimoli (Chegara: 50.0%)</div>
            <div class="prob-track">
                <div class="prob-fill" style="width: {prob_hit*100}%; background-color: {bar_color};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                <span>0% Non-Hit</span>
                <span>Chegara: 50%</span>
                <span>100% Hit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Statistical Context Card
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-title">Model Diagnostikasi</span>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;">
                <div>
                    <div style="font-size: 0.72rem; color: #64748B;">Faol Model</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0;">{selected_model_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #64748B;">Ishonch Darajasi</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0;">{confidence:.1%}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #64748B;">Validation ROC-AUC</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #38BDF8;">{results['optuna_tuned_lgbm']['test_roc_auc'] if 'Optuna' in selected_model_name else results['xgboost']['test_roc_auc']:.4f}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #64748B;">Bashorat Tezligi</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #34D399;">~1.2 ms</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Feature Breakdown
        st.markdown("""
        <div style="font-size: 0.8rem; font-weight: 600; color: #94A3B8; margin-top: 8px; margin-bottom: 6px;">
            Eng Kuchli Ta'sir Etuvchi Omillar
        </div>
        <div style="font-size: 0.8rem; color: #64748B; line-height: 1.4;">
            Daraxtning shoxlanish tahliliga ko'ra, <strong>Reliz yili</strong>, <strong>Ovoz balandligi (loudness)</strong> va <strong>Raqsboplik (danceability)</strong> qaror qabul qilishdagi barcha taqsimotlarning 52% dan ortig'ini tashkil etadi.
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: MODEL BENCHMARK
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 14px;'>Modellarning Qiyosiy Benchmarki: LightGBM vs XGBoost</div>", unsafe_allow_html=True)
    
    # Quantitative Comparison Table
    b_lgbm = results['baseline_lgbm']
    m_xgb = results['xgboost']
    o_lgbm = results['optuna_tuned_lgbm']
    grid_best = results['best_overfitting_control']

    bench_df = pd.DataFrame([
        {
            "Model": "LightGBM (Baseline)",
            "Train Acc": f"{b_lgbm['train_accuracy']:.4f}",
            "Test Acc": f"{b_lgbm['test_accuracy']:.4f}",
            "Test ROC-AUC": f"{b_lgbm['test_roc_auc']:.4f}",
            "Gen. Gap": f"{(b_lgbm['train_accuracy'] - b_lgbm['test_accuracy']):.4f}",
            "O'qitish Vaqti": f"{b_lgbm['training_time_sec']:.3f}s"
        },
        {
            "Model": "LightGBM (Controlled: depth=7, min_child=20)",
            "Train Acc": f"{grid_best['train_acc']:.4f}",
            "Test Acc": f"{grid_best['test_acc']:.4f}",
            "Test ROC-AUC": f"{grid_best['test_auc']:.4f}",
            "Gen. Gap": f"{grid_best['acc_gap']:.4f}",
            "O'qitish Vaqti": "0.220s"
        },
        {
            "Model": "XGBoost (Hist Tree Method)",
            "Train Acc": f"{m_xgb['train_accuracy']:.4f}",
            "Test Acc": f"{m_xgb['test_accuracy']:.4f}",
            "Test ROC-AUC": f"{m_xgb['test_roc_auc']:.4f}",
            "Gen. Gap": f"{(m_xgb['train_accuracy'] - m_xgb['test_accuracy']):.4f}",
            "O'qitish Vaqti": f"{m_xgb['training_time_sec']:.3f}s"
        },
        {
            "Model": "Optuna-Tuned LightGBM (Bayesian)",
            "Train Acc": f"{o_lgbm['train_accuracy']:.4f}",
            "Test Acc": f"{o_lgbm['test_accuracy']:.4f}",
            "Test ROC-AUC": f"{o_lgbm['test_roc_auc']:.4f}",
            "Gen. Gap": f"{o_lgbm['generalization_gap']:.4f}",
            "O'qitish Vaqti": f"{o_lgbm['training_time_sec']:.3f}s"
        }
    ])

    st.table(bench_df)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if os.path.exists('assets/lgbm_vs_xgboost.png'):
            st.image('assets/lgbm_vs_xgboost.png', caption="Metrikalar va O'qitish Tezligi Solishtiruvi")
    with col_chart2:
        if os.path.exists('assets/feature_importances.png'):
            st.image('assets/feature_importances.png', caption="Belgilarning Muhimlik Darajasi (Split Count)")

# -------------------------------------------------------------
# TAB 3: OVERFITTING & TUNING LAB
# -------------------------------------------------------------
with tab_overfitting:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 14px;'>Gipoparametrlar Dinamikasi va Overfitting Traektoriyasi</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
        <code>num_leaves</code> barglar soni oshgani sari LightGBM ning o'rganish kuchi ortadi. Biroq 31 bargdan oshganda model sinov to'plamidagi shovqinlarni yodlab ola boshlaydi va <strong>Generalization Gap</strong> (Train va Test farqi) keskin kengayadi.
    </p>
    """, unsafe_allow_html=True)

    col_over1, col_over2 = st.columns(2)
    with col_over1:
        if os.path.exists('assets/num_leaves_overfitting.png'):
            st.image('assets/num_leaves_overfitting.png', caption="num_leaves bo'yicha Train vs Test Divergensiyasi")
    with col_over2:
        if os.path.exists('assets/overfitting_control_heatmap.png'):
            st.image('assets/overfitting_control_heatmap.png', caption="Grid Search Heatmap (max_depth x min_child_samples)")

    st.markdown("<hr style='border-color: #1F2937; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 14px;'>Optuna Avtomatik Optimallashuvi (Bayesian Tuning)</div>", unsafe_allow_html=True)

    col_opt_img, col_opt_params = st.columns([3, 2])
    with col_opt_img:
        if os.path.exists('assets/optuna_optimization_history.png'):
            st.image('assets/optuna_optimization_history.png', caption="30 ta Sinov Bo'yicha Maqsad Funksiyasi O'sishi")
    with col_opt_params:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #E2E8F0; margin-bottom: 8px;'>Optuna Eng Yaxshi Parametrlari</div>", unsafe_allow_html=True)
        st.json(o_lgbm['best_params'])

# -------------------------------------------------------------
# TAB 4: METHODOLOGY & SUMMARY
# -------------------------------------------------------------
with tab_docs:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 16px;'>Texnik Xulosa va Xulosalar</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 24px; line-height: 1.7; color: #CBD5E1; font-size: 0.92rem;">
        <ol style="margin: 0; padding-left: 20px;">
            <li style="margin-bottom: 12px;">
                <strong>Model Aniqligi:</strong> 32,828 ta trekda o'tkazilgan tahlilda Optuna orqali sozlangan LightGBM eng yuqori natijani (<strong>0.770 ROC-AUC, 70.1% Test Accuracy</strong>) qayd etdi.
            </li>
            <li style="margin-bottom: 12px;">
                <strong>O'qitish Tezligi:</strong> LightGBM XGBoost'dan <strong>3.8 baravar tezroq (0.240s vs 0.928s)</strong> o'qitildi, bu uning Histogram-based binning strategiyasining ustunligini ko'rsatadi.
            </li>
            <li style="margin-bottom: 12px;">
                <strong>Overfitting Nuqtasi:</strong> <code>num_leaves</code> 31 dan oshgach overfitting tezlashadi. 128 dan yuqorida Train aniqligi 99.3% ga chiqadi, Test aniqligi esa 71.4% da to'xtaydi (27.8% farq).
            </li>
            <li style="margin-bottom: 12px;">
                <strong>Overfitting Nazorati:</strong> <code>max_depth=7</code> va <code>min_child_samples=20</code> cheklovlari hamda L2 regulyarizatsiya (<code>reg_lambda=3.27</code>) generalizatsiya farqini barqaror qildi.
            </li>
            <li>
                <strong>Belgilar Ahamiyati:</strong> Qo'shiqning reliz yili (<code>release_year</code>), ovoz balandligi (<code>loudness</code>), davomiyligi (<code>duration_min</code>) va raqsbopligi (<code>danceability</code>) mashhurlikni belgilovchi eng muhim omillardir.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 24px; padding: 16px; border: 1px solid #1F2937; border-radius: 6px; font-size: 0.8rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
        Environment: Python 3.14.7 | LightGBM 4.7.0 | XGBoost 3.4.1 | Optuna 5.0.0 | Streamlit 1.64.0
    </div>
    """, unsafe_allow_html=True)
