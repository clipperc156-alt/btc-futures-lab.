# BTC Futures Lab — read-only MEXC dashboard

Phone-compatible web dashboard for BTC_USDT perpetual futures public market data.
It uses no API key, private account endpoint, or order-placement code.

## What it does
- Reads public MEXC futures ticker and 1-minute candlestick data.
- Retries temporary HTTP errors and handles timeouts.
- Excludes the latest potentially incomplete candle from modelling.
- Estimates whether the close about five minutes later will be higher.
- Reports chronological out-of-sample accuracy and Brier score versus simple baselines.
- Refreshes about every 30 seconds while the app is active.

## Verified MEXC endpoints
- Ticker: `GET https://api.mexc.com/api/v1/contract/ticker?symbol=BTC_USDT`
- Candles: `GET https://api.mexc.com/api/v1/contract/kline/BTC_USDT?interval=Min1`

Official docs:
- https://www.mexc.io/api-docs/futures/market-endpoints/get-ticker-contract-market-data
- https://www.mexc.io/api-docs/futures/market-endpoints/get-candlestick-data
- https://www.mexc.io/api-docs/futures/integration-guide

MEXC documents a maximum of 2,000 candlestick points per request. This app uses the most recent returned points, so the evaluation period is short and not enough to establish a durable edge.

## Deploy free from an iPhone
1. Download and unzip this project.
2. Create/sign in to GitHub in Safari.
3. Create a new repository (public is simplest for this demo; do not add secrets because there are none).
4. Upload `app.py` and `requirements.txt` to the repository root.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Create app → choose your repository → branch `main` → main file path `app.py` → Deploy.
7. Open the deployed URL on your iPhone. Keep the page open for automatic refresh; use “Refresh data now” for manual refresh.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Free-hosting limitation
Streamlit Community Cloud is free, but it is not a guaranteed always-on service. Apps with no traffic for 12 hours can hibernate, and service limits may change. It may also restart or experience upstream/API outages. This is a live dashboard while active, not a guaranteed 24/7 background worker. Do not use third-party uptime pingers to bypass host policies.

## Method and caveats
- Model: StandardScaler + LogisticRegression.
- Features: 1/3/5/10-minute log returns, 10/20-minute rolling volatility, relative volume and candle range.
- Target: whether the close five complete minutes after a feature candle is higher than its close.
- Evaluation: chronological 80/20 split; baseline accuracy is always predicting the majority class from training; Brier baseline uses training-period UP frequency.
- After reporting out-of-sample metrics, a model is fitted on all available labelled data to produce the current estimate.
- Only ~2,000 recent one-minute candles are available in a single request. This is too little for a robust assessment across market regimes.
- The 5-minute estimate is not a MEXC event settlement prediction, and it is not a trading signal or guarantee.
- This dashboard does not place orders, access balances, or need API credentials.

## Troubleshooting
- If the app says an endpoint/response is invalid, inspect MEXC docs or API status; response formats can change.
- If requests are rate-limited or temporarily fail, the app retries selected HTTP errors and will show a message.
- If the app is sleeping, open it and wait for it to wake.
- If deployment fails, open Streamlit's app logs and check that both files are in the repository root.
