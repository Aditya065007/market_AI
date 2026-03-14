import streamlit as st
import pickle
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketAI — Sentiment Intelligence",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Premium CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #050810 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #E8EDF5 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0, 212, 170, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0, 122, 255, 0.08) 0%, transparent 50%),
        #050810 !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Main container ── */
.main .block-container {
    max-width: 680px !important;
    padding: 2rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Hero Section ── */
.hero-wrapper {
    text-align: center;
    padding: 3rem 0 2.5rem;
    animation: fadeSlideDown 0.7s ease both;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.25);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #00D4AA;
    margin-bottom: 1.4rem;
}

.hero-badge .dot {
    width: 6px; height: 6px;
    background: #00D4AA;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 6vw, 3rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 0%, #A8C4E0 50%, #00D4AA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 15px;
    font-weight: 400;
    color: #7A8FAD;
    line-height: 1.6;
    max-width: 420px;
    margin: 0 auto;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
    animation: fadeSlideUp 0.6s ease both;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}

.card-icon {
    width: 36px; height: 36px;
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #C8D8EC;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* ── Textarea override ── */
[data-testid="stTextArea"] label {
    display: none !important;
}

[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 1rem 1.1rem !important;
    resize: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0, 212, 170, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.08) !important;
    outline: none !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #3D5068 !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00D4AA 0%, #007AFF 100%) !important;
    color: #050810 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 24px rgba(0, 212, 170, 0.3) !important;
    margin-top: 0.75rem !important;
}

[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.4) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #5A7090 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}

/* ── Columns gap ── */
[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
}

/* ── Result banners ── */
.result-positive {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.12), rgba(0, 212, 170, 0.05));
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 1rem;
    animation: fadeSlideUp 0.5s ease both;
}

.result-negative {
    background: linear-gradient(135deg, rgba(255, 72, 66, 0.1), rgba(255, 72, 66, 0.04));
    border: 1px solid rgba(255, 72, 66, 0.25);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 1rem;
    animation: fadeSlideUp 0.5s ease both;
}

.result-icon {
    font-size: 28px;
    flex-shrink: 0;
}

.result-text-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.result-text-label.pos { color: #00D4AA; }
.result-text-label.neg { color: #FF4842; }

.result-text-desc {
    font-size: 15px;
    font-weight: 500;
    color: #C8D8EC;
}

/* ── Confidence bar ── */
.conf-bar-wrapper {
    margin-top: 1rem;
}
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #5A7090;
    margin-bottom: 6px;
    font-weight: 500;
}
.conf-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.conf-bar-fill.pos {
    background: linear-gradient(90deg, #00D4AA, #007AFF);
    box-shadow: 0 0 8px rgba(0, 212, 170, 0.5);
}
.conf-bar-fill.neg {
    background: linear-gradient(90deg, #FF4842, #FF8C00);
    box-shadow: 0 0 8px rgba(255, 72, 66, 0.4);
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 1.5rem 0;
}

/* ── Warning ── */
[data-testid="stAlert"] {
    background: rgba(255, 165, 0, 0.08) !important;
    border: 1px solid rgba(255, 165, 0, 0.2) !important;
    border-radius: 14px !important;
    color: #FFB347 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #00D4AA !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 12px;
    color: #2A3A4E;
    letter-spacing: 0.04em;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Configuration ──────────────────────────────────────────────────
MAX_LEN  = 100
LOOKBACK = 5
TICKER   = "^GSPC"

# ── Load Assets ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model("market_model.keras", compile=False)
    with open("tokenizer.pkl",      "rb") as f: tokenizer = pickle.load(f)
    with open("lda_model.pkl",      "rb") as f: lda       = pickle.load(f)
    with open("lda_vectorizer.pkl", "rb") as f: vec       = pickle.load(f)
    with open("scaler_X.pkl",       "rb") as f: scaler_X  = pickle.load(f)
    with open("scaler_y.pkl",       "rb") as f: scaler_y  = pickle.load(f)
    return model, tokenizer, lda, vec, scaler_X, scaler_y

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ts_data():
    df = yf.Ticker(TICKER).history(period="30d")[["Open","High","Low","Close","Volume"]]
    df.dropna(inplace=True)
    return df

# ── Load with spinner ──────────────────────────────────────────────
with st.spinner(""):
    model, tokenizer, lda, vec, scaler_X, scaler_y = load_assets()

# ── Hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">
        <span class="dot"></span>
        Live Market Intelligence
    </div>
    <h1 class="hero-title">Financial Market<br>Sentiment AI</h1>
    <p class="hero-subtitle">
        Paste any financial headline or news article to instantly predict
        S&P 500 direction and market sentiment.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input Card ─────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-header">
        <div class="card-icon">📰</div>
        <span class="card-title">News Input</span>
    </div>
</div>
""", unsafe_allow_html=True)

news_input = st.text_area(
    label="news",
    placeholder="e.g. Federal Reserve signals pause in rate hikes as inflation eases toward 2% target...",
    height=130,
    label_visibility="collapsed"
)

predict_btn = st.button("Analyze Market Sentiment →")

# ── Prediction ─────────────────────────────────────────────────────
if predict_btn:
    if not news_input.strip():
        st.warning("Please enter some financial news text to analyze.")
        st.stop()

    try:
        with st.spinner("Fetching live market data & running model..."):
            # Text
            seq    = tokenizer.texts_to_sequences([news_input])
            X_text = pad_sequences(seq, maxlen=MAX_LEN)

            # Topics
            vec_news = vec.transform([news_input])
            X_topic  = lda.transform(vec_news)

            # Time series
            df_ts = fetch_ts_data()
            if len(df_ts) < LOOKBACK:
                st.error("Insufficient market data. Please try again later.")
                st.stop()

            ts_scaled = scaler_X.transform(df_ts[["Open","High","Low","Close","Volume"]].values)
            X_ts      = ts_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 5)

            # Predict
            reg_pred, cls_pred = model.predict(
                {"text_input": X_text, "ts_input": X_ts, "topic_input": X_topic},
                verbose=0
            )

        price_pred      = float(scaler_y.inverse_transform(reg_pred)[0][0])
        sentiment_score = float(cls_pred[0][0])
        is_positive     = sentiment_score > 0.5
        conf_pct        = round(sentiment_score * 100 if is_positive else (1 - sentiment_score) * 100, 1)
        bar_width       = round(sentiment_score * 100, 1)

        # ── Results ──
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="animation-delay:0.1s">
            <div class="card-header">
                <div class="card-icon">📊</div>
                <span class="card-title">Prediction Results</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted S&P 500 Close", f"${price_pred:,.2f}")
        with col2:
            st.metric("Sentiment Score", f"{round(sentiment_score, 4)}")

        # Confidence bar
        bar_class = "pos" if is_positive else "neg"
        st.markdown(f"""
        <div class="conf-bar-wrapper">
            <div class="conf-bar-label">
                <span>Bearish</span>
                <span>Signal Confidence: {conf_pct}%</span>
                <span>Bullish</span>
            </div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill {bar_class}" style="width:{bar_width}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Result banner
        if is_positive:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-icon">📈</div>
                <div>
                    <div class="result-text-label pos">Bullish Signal Detected</div>
                    <div class="result-text-desc">Model predicts upward market movement</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-icon">📉</div>
                <div>
                    <div class="result-text-label neg">Bearish Signal Detected</div>
                    <div class="result-text-desc">Model predicts downward market movement</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed. Please try again.")
        st.exception(e)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    MarketAI · Powered by CNN-LSTM · S&P 500 data via Yahoo Finance<br>
    Not financial advice · For research purposes only
</div>
""", unsafe_allow_html=True)
