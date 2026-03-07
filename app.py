import time
import requests
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Shark Black Institutional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ESTILO
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #050505 0%, #0a0a0a 50%, #111111 100%);
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090909 0%, #121212 100%);
        border-right: 1px solid rgba(212,175,55,0.16);
    }

    .title-container{
        padding-top: 8px;
        padding-bottom: 18px;
    }

    .title-main{
        font-size: 52px;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(90deg,#FFD700,#F4C430,#D4AF37,#FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 18px rgba(212,175,55,0.22);
        margin-bottom: 4px;
        line-height: 1.0;
    }

    .title-sub{
        font-size: 18px;
        color: #d4af37;
        letter-spacing: 4px;
        font-weight: 700;
        opacity: 0.95;
        margin-top: 4px;
    }

    .gold-line{
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg,transparent,#D4AF37,transparent);
        margin-top: 14px;
        margin-bottom: 6px;
    }

    .gold-card {
        background: linear-gradient(135deg, rgba(25,25,25,0.95), rgba(10,10,10,0.98));
        border: 1px solid rgba(212,175,55,0.25);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 0 18px rgba(212,175,55,0.08);
    }

    .signal-buy {
        color: #00e676;
        font-weight: 900;
        font-size: 24px;
    }

    .signal-sell {
        color: #ff5252;
        font-weight: 900;
        font-size: 24px;
    }

    .signal-neutral {
        color: #ffd54f;
        font-weight: 900;
        font-size: 24px;
    }

    .setup-box {
        background: linear-gradient(135deg, rgba(28,22,8,0.92), rgba(15,12,5,0.98));
        border: 1px solid rgba(242,208,107,0.24);
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 14px;
        color: #f5e6a8;
    }

    .setup-title {
        color: #ffd86b;
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .setup-ok {
        color: #00e676;
        font-weight: 800;
    }

    .setup-wait {
        color: #ffd54f;
        font-weight: 800;
    }

    .setup-bad {
        color: #ff5c5c;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(20,20,20,0.95), rgba(7,7,7,0.98));
        border: 1px solid rgba(212,175,55,0.18);
        padding: 10px 14px;
        border-radius: 16px;
        box-shadow: 0 0 14px rgba(212,175,55,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #c9b36b !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f7f2df !important;
        font-weight: 900 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(212,175,55,0.15);
    }

    .stTabs [data-baseweb="tab"] {
        background: #111111;
        border: 1px solid rgba(212,175,55,0.18);
        border-radius: 12px 12px 0 0;
        color: #e5d08a;
        font-weight: 800;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1c1708, #2a210b);
        color: #ffd86b !important;
        border-color: rgba(212,175,55,0.35);
    }

    .stSelectbox label {
        color: #d7bf73 !important;
        font-weight: 700 !important;
    }

    .block-label {
        color: #f0d982;
        font-weight: 800;
        font-size: 20px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(32,25,8,0.95), rgba(18,14,5,0.98));
        border: 1px solid rgba(212,175,55,0.25);
        border-radius: 16px;
        padding: 14px 16px;
        color: #f6e8b1;
        margin-bottom: 14px;
    }

    .poi-box {
        background: rgba(255, 215, 64, 0.06);
        border: 1px solid rgba(255, 215, 64, 0.18);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
ASSETS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD",
}

ANALYSIS_TFS = ["1h", "15min", "5min"]
EXEC_TF = "5min"

TF_LABELS = {
    "1h": "1H",
    "15min": "15M",
    "5min": "5M",
}

TF_WEIGHTS = {
    "1h": 3.0,
    "15min": 2.0,
    "5min": 1.5,
}

REFRESH_SECONDS = 60

# =========================
# API
# =========================
def get_api_key():
    return st.secrets.get("TWELVE_DATA_API_KEY", None)

@st.cache_data(ttl=20, show_spinner=False)
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
        "close": "Close",
    })

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Datetime").dropna().reset_index(drop=True)
    return df

# =========================
# INDICADORES
# =========================
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

# =========================
# LEITURAS
# =========================
def market_structure(df: pd.DataFrame):
    recent = df.tail(20).copy()
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
    recent = df.tail(20).reset_index(drop=True)
    found = []

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c3 = recent.iloc[i]

        if c3["Low"] > c1["High"]:
            found.append({
                "type": "bullish",
                "top": float(c3["Low"]),
                "bottom": float(c1["High"])
            })

        if c3["High"] < c1["Low"]:
            found.append({
                "type": "bearish",
                "top": float(c1["Low"]),
                "bottom": float(c3["High"])
            })

    return found[-1] if found else None

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

        if candle["Close"] < candle["Open"]:
            if nxt["Close"].iloc[-1] > candle["High"]:
                return {
                    "type": "bullish_ob",
                    "high": float(candle["High"]),
                    "low": float(candle["Low"])
                }

        if candle["Close"] > candle["Open"]:
            if nxt["Close"].iloc[-1] < candle["Low"]:
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

    return {
        "recent_high": float(recent["High"].max()),
        "recent_low": float(recent["Low"].min()),
        "ob": ob,
        "fvg": fvg,
    }

# =========================
# ANÁLISE POR TF
# =========================
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
    rsi14 = float(last["RSI14"])
    atr14 = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else max(price * 0.001, 0.0005)

    structure = market_structure(df)
    sweep = detect_liquidity_sweep(df)
    fvg = detect_fvg(df)
    ifvg = detect_ifvg(df)
    ob = detect_order_block(df)
    amd = detect_amd(df)

    bullish_points = 0
    bearish_points = 0

    if structure == "bullish":
        bullish_points += 2
    elif structure == "bearish":
        bearish_points += 2

    if price > ema20 > ema50:
        bullish_points += 1
    elif price < ema20 < ema50:
        bearish_points += 1

    if rsi14 < 35:
        bullish_points += 1
    elif rsi14 > 65:
        bearish_points += 1

    if sweep == "buy_side_reaction":
        bullish_points += 1
    elif sweep == "sell_side_reaction":
        bearish_points += 1

    if fvg:
        if fvg["type"] == "bullish":
            bullish_points += 1
        elif fvg["type"] == "bearish":
            bearish_points += 1

    if ifvg == "bullish_ifvg":
        bullish_points += 1
    elif ifvg == "bearish_ifvg":
        bearish_points += 1

    if ob:
        if ob["type"] == "bullish_ob":
            bullish_points += 1
        elif ob["type"] == "bearish_ob":
            bearish_points += 1

    if bullish_points - bearish_points >= 2:
        bias = "Bullish"
    elif bearish_points - bullish_points >= 2:
        bias = "Bearish"
    else:
        bias = "Neutral"

    confidence = int(np.clip(55 + abs(bullish_points - bearish_points) * 6, 55, 95))

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
        "bullish_points": bullish_points,
        "bearish_points": bearish_points,
        "bias": bias,
        "confidence": confidence,
        "df": df
    }

# =========================
# TOP-DOWN
# =========================
def classify_trend(results):
    htf = [r for r in results if r["tf_raw"] == "1h"]
    bullish = sum(r["bias"] == "Bullish" for r in htf)
    bearish = sum(r["bias"] == "Bearish" for r in htf)

    if bullish >= 1:
        return "Bullish"
    if bearish >= 1:
        return "Bearish"
    return "Neutral"

def classify_strength(results):
    avg = np.mean([abs(r["bullish_points"] - r["bearish_points"]) for r in results])
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

def apply_fundamental_bias(probability, direction, bias):
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
    weighted_bullish = 0
    weighted_bearish = 0

    for r in results:
        w = TF_WEIGHTS[r["tf_raw"]]
        weighted_bullish += r["bullish_points"] * w
        weighted_bearish += r["bearish_points"] * w

    exec_result = next(r for r in results if r["tf_raw"] == EXEC_TF)
    diff = weighted_bullish - weighted_bearish

    if diff >= 4 and exec_result["bias"] == "Bullish":
        final_direction = "BUY"
    elif diff <= -4 and exec_result["bias"] == "Bearish":
        final_direction = "SELL"
    else:
        final_direction = "NEUTRO — esperando entrada"

    base_probability = int(np.clip(60 + (abs(diff) * 2), 60, 95))

    if final_direction == "BUY":
        base_probability = apply_fundamental_bias(base_probability, "BUY", fundamental_bias)
    elif final_direction == "SELL":
        base_probability = apply_fundamental_bias(base_probability, "SELL", fundamental_bias)

    return final_direction, base_probability

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

# =========================
# PRÉ-ENTRADA / DESCARTE
# =========================
def build_setup(final_direction, exec_result, pois, trend, strength, volatility):
    current_price = exec_result["price"]
    atr_value = exec_result["atr"]

    setup = {
        "status": "Sem setup",
        "why": [],
        "entry_low": None,
        "entry_high": None,
        "ideal_entry": None,
        "invalidation": None,
        "summary": "Sem tese válida no momento.",
        "discarded": False
    }

    if final_direction not in ["BUY", "SELL"]:
        setup["status"] = "Aguardando confluência"
        setup["summary"] = "Os timeframes ainda não alinharam o suficiente para liberar uma entrada."
        setup["why"] = [
            f"Tendência atual: {trend}",
            f"Força do movimento: {strength}",
            f"Volatilidade: {volatility}",
            "O sinal 5M ainda não confirmou a direção final."
        ]
        return setup

    ob = pois["ob"]
    fvg = pois["fvg"]

    if final_direction == "BUY":
        zone_low = None
        zone_high = None

        if ob and ob["type"] == "bullish_ob":
            zone_low = ob["low"]
            zone_high = ob["high"]

        if fvg and fvg["type"] == "bullish":
            if zone_low is None:
                zone_low = fvg["bottom"]
                zone_high = fvg["top"]
            else:
                zone_low = min(zone_low, fvg["bottom"])
                zone_high = max(zone_high, fvg["top"])

        if zone_low is None or zone_high is None:
            zone_low = current_price - atr_value * 0.5
            zone_high = current_price - atr_value * 0.15

        ideal_entry = (zone_low + zone_high) / 2
        invalidation = zone_low - atr_value * 0.35

        if current_price < invalidation:
            setup["status"] = "Setup descartado"
            setup["discarded"] = True
            setup["summary"] = "A hipótese compradora foi invalidada antes da confirmação."
        elif zone_low <= current_price <= zone_high:
            setup["status"] = "Aguardando confirmação na zona"
            setup["summary"] = "O preço entrou na zona compradora. Agora é preciso confirmação de candle."
        elif current_price > zone_high:
            setup["status"] = "Aguardando retração"
            setup["summary"] = "O viés segue comprador, mas a melhor entrada seria em retração até a zona."
        else:
            setup["status"] = "Setup em observação"
            setup["summary"] = "O viés comprador segue válido, aguardando aproximação mais limpa da zona."

        setup["why"] = [
            f"Tendência macro: {trend}",
            f"Força do movimento: {strength}",
            f"Volatilidade: {volatility}",
            f"Estrutura 5M: {exec_result['structure']}",
            "Order Block / FVG comprador como ponto de interesse"
        ]
        setup["entry_low"] = zone_low
        setup["entry_high"] = zone_high
        setup["ideal_entry"] = ideal_entry
        setup["invalidation"] = invalidation
        return setup

    if final_direction == "SELL":
        zone_low = None
        zone_high = None

        if ob and ob["type"] == "bearish_ob":
            zone_low = ob["low"]
            zone_high = ob["high"]

        if fvg and fvg["type"] == "bearish":
            if zone_low is None:
                zone_low = fvg["bottom"]
                zone_high = fvg["top"]
            else:
                zone_low = min(zone_low, fvg["bottom"])
                zone_high = max(zone_high, fvg["top"])

        if zone_low is None or zone_high is None:
            zone_low = current_price + atr_value * 0.15
            zone_high = current_price + atr_value * 0.5

        ideal_entry = (zone_low + zone_high) / 2
        invalidation = zone_high + atr_value * 0.35

        if current_price > invalidation:
            setup["status"] = "Setup descartado"
            setup["discarded"] = True
            setup["summary"] = "A hipótese vendedora foi invalidada antes da confirmação."
        elif zone_low <= current_price <= zone_high:
            setup["status"] = "Aguardando confirmação na zona"
            setup["summary"] = "O preço entrou na zona vendedora. Agora é preciso confirmação de candle."
        elif current_price < zone_low:
            setup["status"] = "Aguardando retração"
            setup["summary"] = "O viés segue vendedor, mas a melhor entrada seria em retração até a zona."
        else:
            setup["status"] = "Setup em observação"
            setup["summary"] = "O viés vendedor segue válido, aguardando aproximação mais limpa da zona."

        setup["why"] = [
            f"Tendência macro: {trend}",
            f"Força do movimento: {strength}",
            f"Volatilidade: {volatility}",
            f"Estrutura 5M: {exec_result['structure']}",
            "Order Block / FVG vendedor como ponto de interesse"
        ]
        setup["entry_low"] = zone_low
        setup["entry_high"] = zone_high
        setup["ideal_entry"] = ideal_entry
        setup["invalidation"] = invalidation
        return setup

    return setup

# =========================
# GRÁFICO
# =========================
def create_candlestick_chart(df, title, setup, stop, tps, pois, final_direction):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles",
        increasing_line_color="#f4d35e",
        decreasing_line_color="#ff5c5c",
        increasing_fillcolor="#f4d35e",
        decreasing_fillcolor="#ff5c5c"
    ))

    fig.add_hline(
        y=pois["recent_high"],
        line_dash="dot",
        line_color="#ffd700",
        annotation_text="Recent High",
        annotation_position="top left"
    )
    fig.add_hline(
        y=pois["recent_low"],
        line_dash="dot",
        line_color="#ff9800",
        annotation_text="Recent Low",
        annotation_position="bottom left"
    )

    if setup["entry_low"] is not None and setup["entry_high"] is not None:
        fig.add_hrect(
            y0=setup["entry_low"],
            y1=setup["entry_high"],
            fillcolor="rgba(0,229,255,0.10)" if final_direction == "BUY" else "rgba(255,82,82,0.10)",
            line_width=1,
            line_color="rgba(255,255,255,0.08)",
            annotation_text="Possível entrada",
            annotation_position="top left"
        )

    if setup["ideal_entry"] is not None:
        fig.add_hline(
            y=setup["ideal_entry"],
            line_color="#00e5ff",
            line_width=2,
            annotation_text="Entrada ideal",
            annotation_position="top left"
        )

    if setup["invalidation"] is not None:
        fig.add_hline(
            y=setup["invalidation"],
            line_color="#ff1744",
            line_width=2,
            annotation_text="Invalidação",
            annotation_position="top left"
        )

    if final_direction in ["BUY", "SELL"] and not setup["discarded"] and tps:
        for i, tp in enumerate(tps, start=1):
            fig.add_hline(
                y=tp,
                line_color="#00e676",
                line_width=1,
                annotation_text=f"TP{i}",
                annotation_position="top right"
            )

    ob = pois["ob"]
    if ob:
        fig.add_hrect(
            y0=ob["low"],
            y1=ob["high"],
            fillcolor="rgba(33, 150, 243, 0.18)" if ob["type"] == "bullish_ob" else "rgba(244, 67, 54, 0.16)",
            line_width=0,
            annotation_text=ob["type"],
            annotation_position="top left"
        )

    fvg = pois["fvg"]
    if fvg:
        fig.add_hrect(
            y0=fvg["bottom"],
            y1=fvg["top"],
            fillcolor="rgba(255, 215, 64, 0.14)" if fvg["type"] == "bullish" else "rgba(255, 112, 67, 0.14)",
            line_width=0,
            annotation_text=f'{fvg["type"]} FVG',
            annotation_position="bottom left"
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b0b0b",
        plot_bgcolor="#0b0b0b",
        font=dict(color="#f5e6a8"),
        xaxis=dict(showgrid=False, rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,215,64,0.06)"),
        height=740,
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode="pan"
    )

    return fig

# =========================
# UI
# =========================
st.markdown("""
<div class="title-container">
    <div class="title-main">SHARK BLACK</div>
    <div class="title-sub">INSTITUTIONAL MARKET INTELLIGENCE</div>
    <div class="gold-line"></div>
</div>
""", unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.error("Chave da API não encontrada nos Secrets.")
    st.stop()

left, right = st.columns([1, 3])

with left:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    asset = st.selectbox("Ativo", list(ASSETS.keys()), index=0)
    chart_tf = st.selectbox("Tempo do gráfico", ANALYSIS_TFS, index=2, format_func=lambda x: TF_LABELS[x])
    fundamental_bias = st.selectbox("Viés fundamental", ["Neutral", "Bullish", "Bearish"], index=0)

    st.markdown('<div class="info-box">Atualização automática ativa • Execução fixa no 5M</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    try:
        results = []
        for tf in ANALYSIS_TFS:
            df = fetch_data(ASSETS[asset], tf, api_key)
            if df.empty:
                st.error(f"Sem dados para {TF_LABELS[tf]}")
                st.stop()
            results.append(analyze_tf(df, tf))

        final_direction, probability = combine_results(results, fundamental_bias)
        exec_result = next(r for r in results if r["tf_raw"] == EXEC_TF)
        chart_result = next(r for r in results if r["tf_raw"] == chart_tf)

        trend = classify_trend(results)
        strength = classify_strength(results)
        volatility = classify_volatility(exec_result)

        pois = poi_levels(exec_result["df"])
        setup = build_setup(final_direction, exec_result, pois, trend, strength, volatility)

        display_direction = final_direction
        if setup["discarded"]:
            display_direction = "NEUTRO — setup descartado"

        trade_direction = "BUY" if final_direction == "BUY" else "SELL" if final_direction == "SELL" else "NEUTRO"
        stop, tps = build_trade(
            setup["ideal_entry"] if setup["ideal_entry"] is not None else exec_result["price"],
            trade_direction if not setup["discarded"] else "NEUTRO",
            exec_result["atr"]
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Preço 5M", f'{exec_result["price"]:.5f}')
        k2.metric("Tendência", trend)
        k3.metric("Força", strength)
        k4.metric("Volatilidade", volatility)
        k5.metric("Probabilidade", f"{probability}%")

        aba1, aba2 = st.tabs(["Painel Principal", "Análise Completa"])

        with aba1:
            st.markdown('<div class="block-label">Resumo do Sinal</div>', unsafe_allow_html=True)

            if display_direction == "BUY":
                st.markdown(f'<div class="signal-buy">Direção final: {display_direction}</div>', unsafe_allow_html=True)
            elif display_direction == "SELL":
                st.markdown(f'<div class="signal-sell">Direção final: {display_direction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-neutral">Direção final: {display_direction}</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="setup-box">
                    <div class="setup-title">Tese antes da entrada</div>
                    <div><b>Status:</b> <span class="{"setup-bad" if "descartado" in setup["status"].lower() else "setup-ok" if "confirmação" in setup["status"].lower() else "setup-wait"}">{setup["status"]}</span></div>
                    <div style="margin-top:8px;"><b>Resumo:</b> {setup["summary"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="block-label">Por que a IA está olhando esse setup</div>', unsafe_allow_html=True)
            for item in setup["why"]:
                st.write(f"- {item}")

            st.markdown('<div class="block-label">Zona sugerida de entrada</div>', unsafe_allow_html=True)
            if setup["entry_low"] is not None and setup["entry_high"] is not None:
                st.write(f"**Zona de entrada:** {setup['entry_low']:.5f} até {setup['entry_high']:.5f}")
                st.write(f"**Entrada ideal:** {setup['ideal_entry']:.5f}")
                st.write(f"**Invalidação:** {setup['invalidation']:.5f}")
            else:
                st.write("Sem zona válida no momento.")

            st.markdown('<div class="block-label">Take Profits</div>', unsafe_allow_html=True)
            if tps and final_direction in ["BUY", "SELL"] and not setup["discarded"]:
                for i, tp in enumerate(tps, start=1):
                    st.write(f"**TP{i}:** {tp:.5f}")
            else:
                st.write("Aguardando confirmação ou setup descartado.")

            st.markdown(f'<div class="block-label">Gráfico em Candles ({TF_LABELS[chart_tf]})</div>', unsafe_allow_html=True)
            fig = create_candlestick_chart(
                chart_result["df"].tail(150),
                f"{asset} - {TF_LABELS[chart_tf]}",
                setup,
                stop,
                tps,
                pois,
                final_direction if not setup["discarded"] else "NEUTRO"
            )
            st.plotly_chart(fig, use_container_width=True)

        with aba2:
            table = pd.DataFrame([
                {
                    "TF": r["tf"],
                    "Viés": r["bias"],
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

            st.markdown('<div class="block-label">Resumo Institucional</div>', unsafe_allow_html=True)
            st.write(f"**Direção final:** {display_direction}")
            st.write(f"**Viés fundamental:** {fundamental_bias}")
            st.write(f"**AMD 5M:** {exec_result['amd']}")
            st.write(f"**Estrutura 5M:** {exec_result['structure']}")
            st.write(f"**Liquidity Sweep 5M:** {exec_result['sweep']}")
            st.write(f"**Order Block 5M:** {exec_result['ob']}")
            st.write(f"**FVG 5M:** {exec_result['fvg']}")
            st.write(f"**IFVG 5M:** {exec_result['ifvg']}")

            st.markdown('<div class="block-label">Pontos de Interesse</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="poi-box">Recent High: {pois["recent_high"]:.5f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="poi-box">Recent Low: {pois["recent_low"]:.5f}</div>', unsafe_allow_html=True)

            if pois["ob"]:
                st.markdown(
                    f'<div class="poi-box">{pois["ob"]["type"]}: {pois["ob"]["low"]:.5f} - {pois["ob"]["high"]:.5f}</div>',
                    unsafe_allow_html=True
                )

            if pois["fvg"]:
                st.markdown(
                    f'<div class="poi-box">{pois["fvg"]["type"]} FVG: {pois["fvg"]["bottom"]:.5f} - {pois["fvg"]["top"]:.5f}</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="block-label">Painel por Timeframe</div>', unsafe_allow_html=True)
            st.dataframe(table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erro ao analisar: {e}")

time.sleep(REFRESH_SECONDS)
st.rerun()
