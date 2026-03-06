import requests
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="IA Forex Pro", layout="wide")

ASSETS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
}

TF_LIST = ["1day", "4h", "1h", "15min", "5min", "1min"]

TF_LABELS = {
    "1day": "1D",
    "4h": "4H",
    "1h": "1H",
    "15min": "15M",
    "5min": "5M",
    "1min": "1M",
}

TF_WEIGHTS = {
    "1day": 3.0,
    "4h": 2.5,
    "1h": 2.0,
    "15min": 1.5,
    "5min": 1.0,
    "1min": 0.8,
}

def get_api_key():
    return st.secrets.get("TWELVE_DATA_API_KEY", None)

def fetch_data(symbol: str, interval: str, apikey: str, outputsize: int = 200) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": apikey,
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if data.get("status") == "error":
        raise ValueError(data.get("message", "Erro ao buscar dados."))

    values = data.get("values")
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df = df.rename(columns={
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    })

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Datetime").dropna().reset_index(drop=True)
    return df

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(length).mean().fillna(method="bfill")

def analyze_tf(df: pd.DataFrame, tf: str):
    df = df.copy()

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)

    last = df.iloc[-1]

    price = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi14 = float(last["RSI14"])
    atr14 = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else max(price * 0.001, 0.0005)

    score = 0

    if ema9 > ema21:
        score += 1
    elif ema9 < ema21:
        score -= 1

    if rsi14 < 35:
        score += 1
    elif rsi14 > 65:
        score -= 1

    if price > ema9:
        score += 1
    elif price < ema9:
        score -= 1

    if score >= 2:
        direction = "BUY"
    elif score <= -2:
        direction = "SELL"
    else:
        direction = "NEUTRO"

    confidence = int(np.clip(60 + abs(score) * 10, 60, 95))

    return {
        "tf": TF_LABELS[tf],
        "tf_raw": tf,
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi14,
        "atr": atr14,
        "score": score,
        "direction": direction,
        "confidence": confidence,
        "df": df
    }

def classify_trend(results):
    htf = [r for r in results if r["tf_raw"] in ["1day", "4h", "1h"]]
    buy = sum(r["direction"] == "BUY" for r in htf)
    sell = sum(r["direction"] == "SELL" for r in htf)

    if buy >= 2:
        return "Alta"
    elif sell >= 2:
        return "Baixa"
    return "Lateral"

def classify_strength(results):
    avg_conf = np.mean([r["confidence"] for r in results])
    if avg_conf >= 85:
        return "Forte"
    elif avg_conf >= 70:
        return "Média"
    return "Fraca"

def classify_volatility(exec_result):
    rel = exec_result["atr"] / max(exec_result["price"], 1e-9)
    if rel >= 0.0012:
        return "Alta"
    elif rel >= 0.0006:
        return "Média"
    return "Baixa"

def combine_results(results):
    weighted_score = 0
    weight_total = 0

    for r in results:
        w = TF_WEIGHTS[r["tf_raw"]]
        weighted_score += r["score"] * w
        weight_total += 3 * w

    norm = weighted_score / weight_total if weight_total else 0

    if norm >= 0.20:
        final_direction = "BUY"
    elif norm <= -0.20:
        final_direction = "SELL"
    else:
        final_direction = "NEUTRO"

    probability = int(np.clip(60 + abs(norm) * 35, 60, 95))

    return final_direction, probability

def build_trade(price, direction, atr_value):
    risk = max(atr_value * 1.5, price * 0.001)

    if direction == "BUY":
        stop = price - risk
        tps = [price + risk * i for i in range(1, 6)]
    elif direction == "SELL":
        stop = price + risk
        tps = [price - risk * i for i in range(1, 6)]
    else:
        stop = price
        tps = []

    return stop, tps

st.title("IA Forex Pro")
st.caption("Análise multi-timeframe com Forex e Ouro. Ferramenta educacional.")

api_key = get_api_key()
if not api_key:
    st.error("Chave da API não encontrada nos Secrets.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    asset = st.selectbox("Ativo", list(ASSETS.keys()), index=0)
    exec_tf = st.selectbox("Período de entrada", ["1min", "5min"], index=1)
    run = st.button("Analisar", use_container_width=True)

with col2:
    if run:
        try:
            results = []
            for tf in TF_LIST:
                df = fetch_data(ASSETS[asset], tf, api_key)
                if df.empty:
                    st.error(f"Sem dados para {TF_LABELS[tf]}")
                    st.stop()
                results.append(analyze_tf(df, tf))

            final_direction, probability = combine_results(results)

            exec_result = next(r for r in results if r["tf_raw"] == exec_tf)

            trend = classify_trend(results)
            strength = classify_strength(results)
            volatility = classify_volatility(exec_result)

            stop, tps = build_trade(exec_result["price"], final_direction, exec_result["atr"])

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Preço", f'{exec_result["price"]:.5f}')
            k2.metric("Tendência", trend)
            k3.metric("Força", strength)
            k4.metric("Volatilidade", volatility)
            k5.metric("Probabilidade", f"{probability}%")

            st.subheader("Sinal Final")
            st.write(f"**Direção:** {final_direction}")
            st.write(f"**Entrada:** {exec_result['price']:.5f}")
            st.write(f"**Stop:** {stop:.5f}")

            st.subheader("Take Profits")
            if tps:
                for i, tp in enumerate(tps, start=1):
                    st.write(f"TP{i}: {tp:.5f}")
            else:
                st.write("Sem take profits.")

            table = pd.DataFrame([
                {
                    "TF": r["tf"],
                    "Direção": r["direction"],
                    "Score": r["score"],
                    "Confiança": f'{r["confidence"]}%',
                    "RSI": round(r["rsi"], 2),
                    "Preço": round(r["price"], 5),
                }
                for r in results
            ])

            st.subheader("Painel por Timeframe")
            st.dataframe(table, use_container_width=True)

            st.subheader(f"Gráfico ({TF_LABELS[exec_tf]})")
            chart_df = results[-1]["df"] if exec_tf == "1min" else next(r["df"] for r in results if r["tf_raw"] == "5min")
            chart_df = chart_df.set_index("Datetime")[["Close"]].tail(200)
            st.line_chart(chart_df)

        except Exception as e:
            st.error(f"Erro ao analisar: {e}")
    else:
        st.info("Clique em Analisar para gerar a análise multi-timeframe.")
