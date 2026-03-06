import requests
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="IA Forex Institucional Pro", layout="wide")

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

def fetch_data(symbol: str, interval: str, apikey: str, outputsize: int = 300) -> pd.DataFrame:
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

    return tr.rolling(length).mean().bfill()

def market_structure(df: pd.DataFrame):
    recent = df.tail(20).copy()
    hh = recent["High"].max()
    ll = recent["Low"].min()
    last = recent.iloc[-1]["Close"]

    ema50 = recent["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = recent["Close"].ewm(span=200, adjust=False).mean().iloc[-1]

    if last > ema50 > ema200:
        return "bullish"
    if last < ema50 < ema200:
        return "bearish"
    return "neutral"

def detect_liquidity_sweep(df: pd.DataFrame):
    recent = df.tail(30).copy()
    if len(recent) < 10:
        return "none"

    prev_high = recent["High"].iloc[:-1].max()
    prev_low = recent["Low"].iloc[:-1].min()

    last = recent.iloc[-1]

    if last["High"] > prev_high and last["Close"] < prev_high:
        return "sell_side_reaction"
    if last["Low"] < prev_low and last["Close"] > prev_low:
        return "buy_side_reaction"
    return "none"

def detect_fvg(df: pd.DataFrame):
    recent = df.tail(10).reset_index(drop=True)
    found = []

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c2 = recent.iloc[i - 1]
        c3 = recent.iloc[i]

        # bullish fvg
        if c3["Low"] > c1["High"]:
            found.append({
                "type": "bullish",
                "top": float(c3["Low"]),
                "bottom": float(c1["High"])
            })

        # bearish fvg
        if c3["High"] < c1["Low"]:
            found.append({
                "type": "bearish",
                "top": float(c1["Low"]),
                "bottom": float(c3["High"])
            })

    if not found:
        return None

    return found[-1]

def detect_ifvg(df: pd.DataFrame):
    fvg = detect_fvg(df)
    if not fvg:
        return None

    last_close = df.iloc[-1]["Close"]

    if fvg["type"] == "bullish" and last_close < fvg["bottom"]:
        return "bearish_ifvg"
    if fvg["type"] == "bearish" and last_close > fvg["top"]:
        return "bullish_ifvg"
    return None

def detect_order_block(df: pd.DataFrame):
    recent = df.tail(25).reset_index(drop=True)
    if len(recent) < 8:
        return None

    for i in range(len(recent) - 4, 1, -1):
        candle = recent.iloc[i]
        nxt = recent.iloc[i + 1:i + 4]

        # bullish OB: last bearish candle before impulse up
        if candle["Close"] < candle["Open"]:
            impulse_up = nxt["Close"].iloc[-1] > candle["High"]
            if impulse_up:
                return {
                    "type": "bullish_ob",
                    "high": float(candle["High"]),
                    "low": float(candle["Low"])
                }

        # bearish OB: last bullish candle before impulse down
        if candle["Close"] > candle["Open"]:
            impulse_down = nxt["Close"].iloc[-1] < candle["Low"]
            if impulse_down:
                return {
                    "type": "bearish_ob",
                    "high": float(candle["High"]),
                    "low": float(candle["Low"])
                }

    return None

def detect_amd(df: pd.DataFrame):
    recent = df.tail(20).copy()
    if len(recent) < 20:
        return "unknown"

    rng = recent["High"].max() - recent["Low"].min()
    body_mean = (recent["Close"] - recent["Open"]).abs().mean()
    last = recent.iloc[-1]

    sweep = detect_liquidity_sweep(df)

    if rng > 0 and body_mean < (rng * 0.10):
        return "Accumulation"

    if sweep != "none":
        return "Manipulation"

    if abs(last["Close"] - recent["Close"].iloc[0]) > (rng * 0.35):
        return "Distribution"

    return "Neutral"

def poi_levels(df: pd.DataFrame):
    recent = df.tail(50)
    ob = detect_order_block(df)
    fvg = detect_fvg(df)

    pois = []
    pois.append(f"Recent High: {recent['High'].max():.5f}")
    pois.append(f"Recent Low: {recent['Low'].min():.5f}")

    if ob:
        pois.append(f"{ob['type']}: {ob['low']:.5f} - {ob['high']:.5f}")

    if fvg:
        pois.append(f"{fvg['type']} FVG: {fvg['bottom']:.5f} - {fvg['top']:.5f}")

    return pois

def analyze_tf(df: pd.DataFrame, tf: str):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)

    last = df.iloc[-1]

    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    rsi14 = float(last["RSI14"])
    atr14 = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else max(price * 0.001, 0.0005)

    structure = market_structure(df)
    sweep = detect_liquidity_sweep(df)
    fvg = detect_fvg(df)
    ifvg = detect_ifvg(df)
    ob = detect_order_block(df)
    amd = detect_amd(df)

    score = 0

    # trend / structure
    if structure == "bullish":
        score += 2
    elif structure == "bearish":
        score -= 2

    # momentum
    if price > ema20 > ema50:
        score += 1
    elif price < ema20 < ema50:
        score -= 1

    # rsi
    if rsi14 < 35:
        score += 1
    elif rsi14 > 65:
        score -= 1

    # liquidity reaction
    if sweep == "buy_side_reaction":
        score += 1
    elif sweep == "sell_side_reaction":
        score -= 1

    # fvg
    if fvg:
        if fvg["type"] == "bullish":
            score += 1
        elif fvg["type"] == "bearish":
            score -= 1

    # ifvg
    if ifvg == "bullish_ifvg":
        score += 1
    elif ifvg == "bearish_ifvg":
        score -= 1

    # order block
    if ob:
        if ob["type"] == "bullish_ob":
            score += 1
        elif ob["type"] == "bearish_ob":
            score -= 1

    if score >= 3:
        direction = "BUY"
    elif score <= -3:
        direction = "SELL"
    else:
        direction = "NEUTRO"

    confidence = int(np.clip(55 + abs(score) * 6, 55, 95))

    return {
        "tf": TF_LABELS[tf],
        "tf_raw": tf,
        "price": price,
        "rsi": rsi14,
        "atr": atr14,
        "structure": structure,
        "sweep": sweep,
        "fvg": fvg["type"] if fvg else "none",
        "ifvg": ifvg if ifvg else "none",
        "ob": ob["type"] if ob else "none",
        "amd": amd,
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
        return "Bullish"
    if sell >= 2:
        return "Bearish"
    return "Neutral"

def classify_strength(results):
    avg = np.mean([abs(r["score"]) for r in results])
    if avg >= 4:
        return "Strong"
    if avg >= 2.5:
        return "Medium"
    return "Weak"

def classify_volatility(exec_result):
    rel = exec_result["atr"] / max(exec_result["price"], 1e-9)
    if rel >= 0.0012:
        return "High"
    if rel >= 0.0006:
        return "Medium"
    return "Low"

def apply_fundamental_bias(direction, probability, bias):
    if bias == "Bullish":
        if direction == "BUY":
            probability = min(95, probability + 5)
        elif direction == "SELL":
            probability = max(55, probability - 5)

    elif bias == "Bearish":
        if direction == "SELL":
            probability = min(95, probability + 5)
        elif direction == "BUY":
            probability = max(55, probability - 5)

    return probability

def combine_results(results, fundamental_bias):
    weighted_score = 0
    weight_total = 0

    for r in results:
        w = TF_WEIGHTS[r["tf_raw"]]
        weighted_score += r["score"] * w
        weight_total += 6 * w

    norm = weighted_score / weight_total if weight_total else 0

    if norm >= 0.20:
        final_direction = "BUY"
    elif norm <= -0.20:
        final_direction = "SELL"
    else:
        final_direction = "NEUTRO"

    probability = int(np.clip(60 + abs(norm) * 40, 60, 95))
    probability = apply_fundamental_bias(final_direction, probability, fundamental_bias)

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

st.title("IA Forex Institucional Pro")
st.caption("Top-down + OB + FVG + IFVG + AMD + POI + viés fundamental. Ferramenta educacional.")

api_key = get_api_key()
if not api_key:
    st.error("Chave da API não encontrada nos Secrets.")
    st.stop()

left, right = st.columns([1, 3])

with left:
    asset = st.selectbox("Ativo", list(ASSETS.keys()), index=0)
    exec_tf = st.selectbox("Período de entrada", ["1min", "5min"], index=1)
    fundamental_bias = st.selectbox(
        "Viés fundamental",
        ["Neutral", "Bullish", "Bearish"],
        index=0
    )
    run = st.button("Analisar", use_container_width=True)

with right:
    if run:
        try:
            results = []
            for tf in TF_LIST:
                df = fetch_data(ASSETS[asset], tf, api_key)
                if df.empty:
                    st.error(f"Sem dados para {TF_LABELS[tf]}")
                    st.stop()
                results.append(analyze_tf(df, tf))

            final_direction, probability = combine_results(results, fundamental_bias)
            exec_result = next(r for r in results if r["tf_raw"] == exec_tf)

            trend = classify_trend(results)
            strength = classify_strength(results)
            volatility = classify_volatility(exec_result)

            stop, tps = build_trade(exec_result["price"], final_direction, exec_result["atr"])
            pois = poi_levels(exec_result["df"])

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Preço", f'{exec_result["price"]:.5f}')
            k2.metric("Tendência", trend)
            k3.metric("Força", strength)
            k4.metric("Volatilidade", volatility)
            k5.metric("Probabilidade", f"{probability}%")

            st.subheader("Resumo Institucional")
            st.write(f"**Direção final:** {final_direction}")
            st.write(f"**Viés fundamental:** {fundamental_bias}")
            st.write(f"**AMD:** {exec_result['amd']}")
            st.write(f"**Estrutura:** {exec_result['structure']}")
            st.write(f"**Liquidity Sweep:** {exec_result['sweep']}")
            st.write(f"**Order Block:** {exec_result['ob']}")
            st.write(f"**FVG:** {exec_result['fvg']}")
            st.write(f"**IFVG:** {exec_result['ifvg']}")

            st.subheader("Entrada")
            st.write(f"**Entrada:** {exec_result['price']:.5f}")
            st.write(f"**Stop:** {stop:.5f}")

            st.subheader("Take Profits")
            if tps:
                for i, tp in enumerate(tps, start=1):
                    st.write(f"TP{i}: {tp:.5f}")
            else:
                st.write("Sem take profits.")

            st.subheader("Pontos de Interesse")
            for p in pois:
                st.write(f"- {p}")

            table = pd.DataFrame([
                {
                    "TF": r["tf"],
                    "Direção": r["direction"],
                    "Score": r["score"],
                    "Confiança": f'{r["confidence"]}%',
                    "Estrutura": r["structure"],
                    "Sweep": r["sweep"],
                    "OB": r["ob"],
                    "FVG": r["fvg"],
                    "IFVG": r["ifvg"],
                    "AMD": r["amd"],
                    "RSI": round(r["rsi"], 2),
                    "Preço": round(r["price"], 5),
                }
                for r in results
            ])

            st.subheader("Painel por Timeframe")
            st.dataframe(table, use_container_width=True)

            st.subheader(f"Gráfico ({TF_LABELS[exec_tf]})")
            chart_df = next(r["df"] for r in results if r["tf_raw"] == exec_tf)
            chart_df = chart_df.set_index("Datetime")[["Close"]].tail(200)
            st.line_chart(chart_df)

        except Exception as e:
            st.error(f"Erro ao analisar: {e}")
    else:
        st.info("Clique em Analisar para gerar a leitura institucional.")
