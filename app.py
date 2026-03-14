import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Configuration
# -------------------------------

MAX_LEN = 100   # must match training length


# -------------------------------
# Load ML Assets
# -------------------------------

@st.cache_resource
def load_assets():

    model = load_model("market_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)

    with open("lda_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    return model, tokenizer, lda, vec


model, tokenizer, lda, vec = load_assets()


# -------------------------------
# UI
# -------------------------------

st.title("📊 Financial Market Intelligence")

st.write(
    "Predict potential market sentiment from financial news using an AI model."
)

news_input = st.text_area("Enter Financial News Text")


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Market Sentiment"):

    if not news_input.strip():
        st.warning("Please enter some news text.")
        st.stop()

    try:

        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([news_input])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN)

        # LDA topic features
        vec_news = vec.transform([news_input])
        topic_features = lda.transform(vec_news)

        # Prediction
        prediction = model.predict([padded_seq, topic_features])

        score = float(prediction[0][0])

        st.subheader("Prediction Result")
        st.write("Confidence:", round(score, 4))

        if score > 0.5:
            st.success("📈 Positive Market Sentiment")
        else:
            st.error("📉 Negative Market Sentiment")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
