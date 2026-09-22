# MarketAI — Multi-Modal Market Sentiment Predictor

**Live App:** https://marketai-vzbkkbt7amtfoqvt3pzowz.streamlit.app/

A Streamlit app that combines three very different kinds of data — financial news text, recent price history, and topic modeling — into a single neural network that predicts next-day S&P 500 closing price and market sentiment direction. This project is a research and architecture exercise in fusing multiple data modalities into one model; it does not report a validated accuracy figure, and its predictions should be read as illustrative of the technique rather than as financial advice.

## Features

- **Multi-modal input fusion** — combines three distinct data types into one prediction: news headline text, historical price/volume data, and topic-model output
- **Dual-task output** — a single fused representation feeds two separate output heads at once: one predicting a price (regression), one predicting sentiment direction (classification)
- **Live data ingestion** — pulls recent market data at the moment of prediction via `yfinance`, rather than relying on a static, pre-downloaded dataset

## Tech Stack, and why each piece is here

- **TensorFlow / Keras** — builds and runs the neural network itself, including the multi-branch architecture described below.
- **yfinance** — a library for pulling live stock market data (price, volume, etc.) directly from Yahoo Finance, used here to fetch the most recent 30 days of trading data at prediction time.
- **gensim** — used to train the LDA (Latent Dirichlet Allocation) topic model, a classic technique for discovering the underlying "topics" present across a collection of text without being told what those topics are in advance.
- **NLTK** — general text preprocessing support for the topic modeling and text pipeline.
- **BeautifulSoup / lxml** — used during the data-collection stage (not in the deployed app itself) to parse news content pulled from the web.
- **Streamlit** — the app interface.

## Architecture — three branches, one prediction

The model doesn't process all its inputs the same way — each data type gets handled by the type of neural network layer best suited to it, and the results are combined at the end:

- **Text branch:** a news headline is first converted into a sequence of 100 tokens, then passed through a *frozen* pretrained GloVe embedding layer (10,000-word vocabulary, 50 dimensions). "Frozen" means these word representations were learned elsewhere on a huge amount of text and are used as-is here, rather than retrained from scratch. From there, a Conv1D layer (128 filters) scans across the sequence looking for useful word patterns, and a global max-pooling step condenses that into one summary vector for the headline.
- **Time-series branch:** the last 5 days of OHLCV data (Open, High, Low, Close, Volume) are fed into an LSTM (Long Short-Term Memory) layer with 64 units — LSTMs are specifically designed to pick up on patterns across a sequence of steps over time, which is what a short price history is.
- **Topic branch:** the LDA topic vector (which topics the current news falls under, and how strongly) is passed through a small Dense layer (16 units) to bring it into the same kind of representation as the other two branches.

These three summary vectors — one from text, one from price history, one from topics — are concatenated together and passed through a shared Dense(64) layer with Dropout, and then split into two separate output heads: a linear-activation head that outputs a predicted price, and a sigmoid-activation head that outputs a bullish/bearish probability.

In total the model has about 652,000 parameters, though roughly 500,000 of those belong to the frozen embedding layer and aren't actually updated during training — the number of parameters the model is genuinely learning from data is closer to 150,000.

