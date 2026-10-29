import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

# READ-ONLY: this app uses public market-data GET endpoints only.
# It has no API keys, account access, or order-placement code.

st.set_page_config(page_title="BTC Futures Lab", page_icon="₿", layout="wide")
BASE_URL = "https://api.mexc.com"
SYMBOL = "BTC_USDT"
INTERVAL = "Min1"
REFRESH_SECONDS = 30
HORIZON = 5

@st.cache_resource
def http_session():
    session = requests.Session()
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "BTC-Futures-Lab/1.0 (read-only public market data)"})
    return session

def get_json(path, params=None):
    response = http_session().get(BASE_URL + path, params=params, timeout=(5, 15))
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(f"MEXC API returned an error: {payload.get('message', payload.get('code'))}")
    return payload

@st.cache_data(ttl=20, show_spinner=False)
def fetch_ticker():
    payload = get_json("/api/v1/contract/ticker", {"symbol": SYMBOL})
    data = payload.get("data")
    if isinstance(data, list):
        data = next((x for x in data if x.get("symbol") == SYMBOL), None)
    if not isinstance(data, dict) or "lastPrice" not in data:
        raise RuntimeError("Unexpected ticker response; cannot find BTC_USDT lastPrice.")
    return data

@st.cache_data(ttl=20, show_spinner=False)
def fetch_candles():
    payload = get_json(f"/api/v1/contract/kline/{SYMBOL}", {"interval": INTERVAL})
    data = payload.get("data")
    if not isinstance(data, dict) or not all(k in data for k in ("time", "open", "high", "low", "close", "vol")):
        raise RuntimeError("Unexpected candlestick response from MEXC.")
    df = pd.DataFrame({
        "time": pd.to_datetime(data["time"], unit="s", utc=True),
        "open": pd.to_numeric(data["open"], errors="coerce"),
        "high": pd.to_numeric(data["high"], errors="coerce"),
        "low": pd.to_numeric(data["low"], errors="coerce"),
        "close": pd.to_numeric(data["close"], errors="coerce"),
        "volume": pd.to_numeric(data["vol"], errors="coerce"),
    })
    df = df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    # The latest returned minute may still be forming. Exclude it from modelling.
    if len(df) > 2:
        df = df.iloc[:-1].copy()
    if len(df) < 300:
        raise RuntimeError(f"Only {len(df)} closed candles returned; need at least 300.")
    return df

def make_features(raw):
    x = raw.copy()
    log_close = np.log(x["close"].clip(lower=1e-9))
    for n in (1, 3, 5, 10):
        x[f"ret_{n}"] = log_close.diff(n)
    x["volatility_10"] = x["ret_1"].rolling(10).std()
    x["volatility_20"] = x["ret_1"].rolling(20).std()
    x["volume_ratio"] = x["volume"] / x["volume"].rolling(20).mean().replace(0, np.nan)
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    # At time t, predict whether the close five complete minutes later is higher.
    x["target_up_5m"] = (x["close"].shift(-HORIZON) > x["close"]).astype(float)
    x.loc[x.index[-HORIZON:], "target_up_5m"] = np.nan
    cols = ["ret_1", "ret_3", "ret_5", "ret_10", "volatility_10",
            "volatility_20", "volume_ratio", "range_pct"]
    ready = x.dropna(subset=cols + ["target_up_5m"]).copy()
    ready["target_up_5m"] = ready["target_up_5m"].astype(int)
    return ready, cols

def evaluate_and_fit(raw):
    data, features = make_features(raw)
    if len(data) < 250:
        raise RuntimeError(f"Not enough usable observations after feature creation: {len(data)}.")
    split = int(len(data) * 0.80)
    train, test = data.iloc[:split], data.iloc[split:]
    if train["target_up_5m"].nunique() < 2 or test["target_up_5m"].nunique() < 2:
        raise RuntimeError("The train/test period has only one target class. Try again later.")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000, class_weight=None, random_state=42)),
    ])
    model.fit(train[features], train["target_up_5m"])
    probs = model.predict_proba(test[features])[:, list(model.named_steps["logistic"].classes_).index(1)]
    preds = (probs >= 0.5).astype(int)
    majority = int(train["target_up_5m"].mean() >= 0.5)
    baseline_preds = np.full(len(test), majority, dtype=int)
    # Brier baseline uses the UP frequency in training data, not test labels.
    train_up_rate = float(train["target_up_5m"].mean())
    metrics = {
        "observations": len(data),
        "train_n": len(train),
        "test_n": len(test),
        "accuracy": float(accuracy_score(test["target_up_5m"], preds)),
        "majority_accuracy": float(accuracy_score(test["target_up_5m"], baseline_preds)),
        "brier": float(brier_score_loss(test["target_up_5m"], probs)),
        "brier_baseline": float(brier_score_loss(test["target_up_5m"], np.full(len(test), train_up_rate))),
        "test_up_rate": float(test["target_up_5m"].mean()),
        "train_up_rate": train_up_rate,
        "test_start": test["time"].iloc[0],
        "test_end": test["time"].iloc[-1],
    }
    # Refit on all available labelled observations for the current live estimate.
    live_model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    live_model.fit(data[features], data["target_up_5m"])
    latest = data.iloc[[-1]]
    classes = list(live_model.named_steps["logistic"].classes_)
    up_prob = float(live_model.predict_proba(latest[features])[0, classes.index(1)])
    return metrics, up_prob, data, features

st.title("₿ BTC Futures Lab")
st.caption("Read-only BTC_USDT perpetual market analysis · 5-minute direction estimate")
st.info("Educational research only. Probabilities are uncertain statistical estimates, not trading instructions. This app cannot place orders and does not use exchange API keys.")

# Auto-refresh the page while open. Hosting can still sleep/restart under free-tier rules.
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="btc_refresh")
except Exception:
    st.warning("Auto-refresh package unavailable. Use the Refresh data button.")

left, right = st.columns([1, 5])
with left:
    if st.button("Refresh data now", use_container_width=True):
        fetch_ticker.clear()
        fetch_candles.clear()
        st.rerun()

try:
    ticker = fetch_ticker()
    candles = fetch_candles()
    metrics, up_prob, data, feature_cols = evaluate_and_fit(candles)
except Exception as exc:
    st.error(f"Market data/model error: {exc}")
    st.caption("The app will retry on the next refresh. Check MEXC API status and the app logs if this persists.")
    st.stop()

price = float(ticker["lastPrice"])
bid = ticker.get("bid1")
ask = ticker.get("ask1")
index_price = ticker.get("indexPrice")
fair_price = ticker.get("fairPrice")
api_ms = ticker.get("timestamp")
api_time = datetime.fromtimestamp(float(api_ms) / 1000, tz=timezone.utc) if api_ms else None
data_age = (datetime.now(timezone.utc) - api_time).total_seconds() if api_time else None

if data_age is not None and data_age > 180:
    st.warning(f"Ticker timestamp is about {data_age:.0f} seconds old. Treat the live estimate as stale until data refreshes.")

direction = "UP-LEANING" if up_prob >= 0.55 else "DOWN-LEANING" if up_prob <= 0.45 else "UNCERTAIN"
confidence = max(up_prob, 1 - up_prob)
if confidence < 0.55:
    direction = "UNCERTAIN"

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC futures last price", f"${price:,.2f}")
m2.metric("5-min UP probability", f"{up_prob:.1%}")
m3.metric("5-min DOWN probability", f"{1-up_prob:.1%}")
m4.metric("Model reading", direction)

with st.expander("Market details"):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Best bid", f"${float(bid):,.2f}" if bid is not None else "Unavailable")
    d2.metric("Best ask", f"${float(ask):,.2f}" if ask is not None else "Unavailable")
    d3.metric("Index price", f"${float(index_price):,.2f}" if index_price is not None else "Unavailable")
    d4.metric("Fair price", f"${float(fair_price):,.2f}" if fair_price is not None else "Unavailable")
    st.write("Ticker timestamp (UTC):", api_time.isoformat() if api_time else "Unavailable")
    st.write("Closed candles used:", len(candles), " | Latest closed candle (UTC):", candles["time"].iloc[-1].isoformat())

st.subheader("Recent BTC futures candles")
chart_df = candles.tail(180)
fig = go.Figure(data=[go.Candlestick(
    x=chart_df["time"], open=chart_df["open"], high=chart_df["high"],
    low=chart_df["low"], close=chart_df["close"], name="BTC_USDT"
)])
fig.update_layout(height=430, xaxis_title="Time (UTC)", yaxis_title="Price (USDT)",
                  xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Out-of-sample evaluation")
st.caption("Chronological 80/20 split on available labelled candles. The final 20% is evaluated before fitting the live model on all available labelled data. The test period is short and may not represent other market regimes.")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Test accuracy", f"{metrics['accuracy']:.1%}", help="Fraction of test predictions classified correctly.")
e2.metric("Majority baseline", f"{metrics['majority_accuracy']:.1%}", help="Always predicts the more frequent direction in the training period.")
e3.metric("Brier score", f"{metrics['brier']:.4f}", help="Lower is better; measures probability error.")
e4.metric("Training-rate Brier baseline", f"{metrics['brier_baseline']:.4f}", help="Constant UP probability based only on the training set.")
st.write(
    f"Usable labelled observations: **{metrics['observations']}** · "
    f"Train: **{metrics['train_n']}** · Test: **{metrics['test_n']}**"
)
st.write(
    f"Test period: **{metrics['test_start'].strftime('%Y-%m-%d %H:%M UTC')}** to "
    f"**{metrics['test_end'].strftime('%Y-%m-%d %H:%M UTC')}** · "
    f"UP frequency in test: **{metrics['test_up_rate']:.1%}**"
)
if metrics["accuracy"] <= metrics["majority_accuracy"]:
    st.warning("The model did not beat the majority-class accuracy baseline on this test period.")
if metrics["brier"] >= metrics["brier_baseline"]:
    st.warning("The model's probability score did not beat the training-frequency Brier baseline on this test period.")

st.subheader("Latest estimate details")
st.write(f"Estimated probability that the closed price about five minutes after the latest feature candle is higher: **{up_prob:.1%} UP / {1-up_prob:.1%} DOWN**.")
st.write("This target is a five-minute close-to-close direction estimate, not MEXC event settlement and not a guaranteed price target.")
st.write("Features: 1/3/5/10-minute returns, 10/20-minute volatility, relative volume and candle range. Model: scaled logistic regression.")
st.caption(f"Dashboard refresh target: every {REFRESH_SECONDS} seconds while the app is active. Last dashboard render (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}.")
st.caption("No private endpoints, API keys, trading permissions, order placement, or account data are used.")
