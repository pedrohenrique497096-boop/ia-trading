import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IA Forex Simples", layout="wide")

ASSETS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "XAUUSD": "XAUUSD=X",
    "USDJPY": "JPY=X",
}

TIMEFRAMES = {
    "1M": ("1d", "1m"),
    "5M": ("5d", "5m"),
    "15M": ("10d", "15m"),
    "1H": ("1mo", "60m"),
    "1D": ("6mo", "1d"),
}

def get_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

def analyze(df: pd.DataFrame):
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    last = df.iloc[-1]
    price = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])

    if ema9 > ema21:
        direction = "BUY"
        confidence = 72
    elif ema9 < ema21:
        direction = "SELL"
        confidence = 72
    else:
        direction = "NEUTRO"
        confidence = 50

    risk = max(price * 0.001, 0.0005)

    if direction == "BUY":
        stop = price - risk
        tps = [price + risk * i for i in range(1, 6)]
    elif direction == "SELL":
        stop = price + risk
        tps = [price - risk * i for i in range(1, 6)]
    else:
        stop = price
        tps = []

    return price, direction, confidence, stop, tps

st.title("IA Forex Simples")
st.caption("Ferramenta educacional. Não é recomendação financeira.")

col1, col2 = st.columns([1, 2])

with col1:
    asset = st.selectbox("Ativo", list(ASSETS.keys()), index=0)
    tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1)
    run = st.button("Analisar", use_container_width=True)

with col2:
    if run:
        period, interval = TIMEFRAMES[tf]
        df = get_data(ASSETS[asset], period, interval)

        if df.empty:
            st.error("Não consegui puxar os dados agora.")
        else:
            price, direction, confidence, stop, tps = analyze(df)

            a, b, c = st.columns(3)
            a.metric("Preço", f"{price:.5f}")
            b.metric("Direção", direction)
            c.metric("Confiança", f"{confidence}%")

            st.write(f"**Stop:** {stop:.5f}")

            st.subheader("Take Profits")
            if tps:
                for i, tp in enumerate(tps, start=1):
                    st.write(f"TP{i}: {tp:.5f}")
            else:
                st.write("Sem take profits.")

            st.subheader("Gráfico")
            st.line_chart(df[["Close"]].tail(200))
    else:
        st.info("Clique em Analisar para gerar a análise.")
