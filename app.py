import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IA Forex Simples", layout="wide")

ASSETS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
}

TIMEFRAMES = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h": "1h",
    "1day": "1day",
}

def get_api_key():
    if "TWELVE_DATA_API_KEY" in st.secrets:
        return st.secrets["TWELVE_DATA_API_KEY"]
    return None

def get_data(symbol: str, interval: str, apikey: str) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": apikey,
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if "status" in data and data["status"] == "error":
        raise ValueError(data.get("message", "Erro ao buscar dados."))

    values = data.get("values")
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df = df.rename(columns={"datetime": "Datetime", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Datetime").dropna().reset_index(drop=True)
    return df

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

api_key = get_api_key()

if not api_key:
    st.error("Chave da API não encontrada. Adicione TWELVE_DATA_API_KEY nos Secrets do Streamlit.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    asset = st.selectbox("Ativo", list(ASSETS.keys()), index=0)
    tf = st.selectbox("Período de tempo", list(TIMEFRAMES.keys()), index=1)
    run = st.button("Analisar", use_container_width=True)

with col2:
    if run:
        try:
            df = get_data(ASSETS[asset], TIMEFRAMES[tf], api_key)

            if df.empty:
                st.error("A API retornou vazio para esse ativo/timeframe.")
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

                st.subheader("Últimos dados")
                st.dataframe(df.tail(10), use_container_width=True)

                st.subheader("Gráfico")
                chart_df = df.set_index("Datetime")[["Close"]].tail(200)
                st.line_chart(chart_df)
        except Exception as e:
            st.error(f"Erro ao analisar: {e}")
    else:
        st.info("Clique em Analisar para gerar a análise.")
