import streamlit as st
import pickle
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Configuration
# -------------------------------
MAX_LEN  = 100
LOOKBACK = 5
TICKER   = "^GSPC"  # S&P 500

# -------------------------------
# Load ML Assets
# -------------------------------
@st.cache_resource
def load_assets():
    model = load_model("market_model.keras", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
    with open("lda_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return model, tokenizer, lda, vec, scaler_X, scaler_y

model, tokenizer, lda, vec, scaler_X, scaler_y = load_assets()

# -------------------------------
# Fetch S&P 500 OHLCV data
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_ts_data():
    df = yf.Ticker(TICKER).history(period="30d")[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

# -------------------------------
# UI
# -------------------------------
st.title("📊 Financial Market Intelligence")
st.write("Predict market sentiment and next-day S&P 500 direction from financial news.")

news_input = st.text_area("Enter Financial News Text")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Market Sentiment"):
    if not news_input.strip():
        st.warning("Please enter some news text.")
        st.stop()

    try:
        # --- Text input ---
        seq    = tokenizer.texts_to_sequences([news_input])
        X_text = pad_sequences(seq, maxlen=MAX_LEN)

        # --- Topic input ---
        vec_news = vec.transform([news_input])
        X_topic  = lda.transform(vec_news)

        # --- Time series input ---
        with st.spinner("Fetching latest S&P 500 data..."):
            df_ts = fetch_ts_data()

        if len(df_ts) < LOOKBACK:
            st.error("Not enough market data available. Try again later.")
            st.stop()

        ts_scaled = scaler_X.transform(df_ts[["Open", "High", "Low", "Close", "Volume"]].values)
        X_ts      = ts_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 5)

        # --- Predict ---
        reg_pred, cls_pred = model.predict(
            {"text_input": X_text, "ts_input": X_ts, "topic_input": X_topic},
            verbose=0
        )

        price_pred      = float(scaler_y.inverse_transform(reg_pred)[0][0])
        sentiment_score = float(cls_pred[0][0])

        # --- Display results ---
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Next-Day S&P 500 Close", f"${price_pred:,.2f}")
        with col2:
            st.metric("Sentiment Confidence", f"{round(sentiment_score, 4)}")

        if sentiment_score > 0.5:
            st.success("📈 Positive Market Sentiment — Predicted UP Movement")
        else:
            st.error("📉 Negative Market Sentiment — Predicted DOWN Movement")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
