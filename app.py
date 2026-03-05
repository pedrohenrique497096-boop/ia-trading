import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Vortex-like Forex AI (Top-Down)", layout="wide")

ASSETS = {
    "XAUUSD": "XAUUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
}

TF_CHAIN = ["1D", "4H", "1H", "15M", "5M", "1M"]

TF_FETCH = {
    "1D": ("6mo", "1d"),
    "1H": ("1mo", "60m"),
    "15M": ("10d", "15m"),
    "5M": ("5d", "5m"),
    "1M": ("1d", "1m"),
}

TF_WEIGHTS = {"1D": 3.0, "4H": 2.5, "1H": 2.0, "15M": 1.5, "5M": 1.0, "1M": 0.8}

# ----------------------------
# Indicators
# ----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean().fillna(method="bfill")

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(length).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(length).mean() / atr_val)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(length).mean() / atr_val)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(length).mean().fillna(method="bfill").fillna(15)

# ----------------------------
# Data
# ----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

def aggregate_to_4h_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    o = df_1h["Open"].resample("4H").first()
    h = df_1h["High"].resample("4H").max()
    l = df_1h["Low"].resample("4H").min()
    c = df_1h["Close"].resample("4H").last()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    return out

# ----------------------------
# Analysis per timeframe
# ----------------------------
@dataclass
class TFResult:
    tf: str
    direction: str          # BUY/SELL/NEUTRO
    score: int              # -3..+3
    confidence: int         # 50..95
    close: float
    ema20: float
    ema50: float
    rsi14: float
    adx14: float
    atr14: float

def analyze_tf(df: pd.DataFrame, tf: str) -> TFResult:
    df = df.copy()
    close = df["Close"]
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["RSI14"] = rsi(close, 14)
    df["ADX14"] = adx(df, 14)
    df["ATR14"] = atr(df, 14)

    last = df.iloc[-1]
    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi14 = float(last["RSI14"])
    adx14 = float(last["ADX14"])
    atr14 = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else max(price * 0.0005, 1e-6)

    score = 0
    # Trend via EMAs
    if ema20 > ema50:
        score += 1
    elif ema20 < ema50:
        score -= 1

    # Momentum price vs EMA20
    if price > ema20:
        score += 1
    elif price < ema20:
        score -= 1

    # RSI tilt
    if rsi14 < 35:
        score += 1
    elif rsi14 > 65:
        score -= 1

    # Confidence from score + ADX (strength)
    base_conf = 55 + min(abs(score) * 12, 36)
    if adx14 < 18:
        base_conf -= 8
    elif adx14 > 25:
        base_conf += 4
    conf = int(np.clip(base_conf, 50, 95))

    if score >= 2:
        direction = "BUY"
    elif score <= -2:
        direction = "SELL"
    else:
        direction = "NEUTRO"

    return TFResult(
        tf=tf,
        direction=direction,
        score=int(score),
        confidence=conf,
        close=price,
        ema20=ema20,
        ema50=ema50,
        rsi14=rsi14,
        adx14=adx14,
        atr14=atr14,
    )

# ----------------------------
# Top-down combine + Vortex-like metrics
# ----------------------------
def combine_top_down(results: List[TFResult]) -> Tuple[str, int, float, int, int]:
    """
    Retorna:
    - direção_final (BUY/SELL/NEUTRO)
    - probabilidade_final (55..95)
    - norm_score (-1..+1)
    - alinhamento_percent (0..100)
    - conflito (0..N)
    """
    weighted = 0.0
    max_abs = 0.0

    for r in results:
        w = TF_WEIGHTS.get(r.tf, 1.0)
        weighted += w * r.score
        max_abs += w * 3.0

    norm = 0.0 if max_abs == 0 else (weighted / max_abs)  # -1..+1

    if norm >= 0.20:
        direction = "BUY"
    elif norm <= -0.20:
        direction = "SELL"
    else:
        direction = "NEUTRO"

    # Probabilidade (estilo Vortex): depende do alinhamento
    base_prob = 60 + abs(norm) * 35  # 60..95

    dirs = [r.direction for r in results if r.direction != "NEUTRO"]
    buy_ct = sum(d == "BUY" for d in dirs)
    sell_ct = sum(d == "SELL" for d in dirs)
    conflict = min(buy_ct, sell_ct)

    # alinhamento simples: % dos TFs que concordam com a direção final
    if direction == "BUY":
        agree = sum(r.direction == "BUY" for r in results)
    elif direction == "SELL":
        agree = sum(r.direction == "SELL" for r in results)
    else:
        agree = sum(r.direction == "NEUTRO" for r in results)
    alignment_pct = int(round(100 * agree / len(results)))

    # penaliza conflito
    if conflict >= 2:
        base_prob -= 10
    elif conflict == 1:
        base_prob -= 5

    prob = int(np.clip(base_prob, 55, 95))
    return direction, prob, norm, alignment_pct, conflict

def classify_trend(results: List[TFResult]) -> Tuple[str, str]:
    """
    Tendência baseada em HTF (1D/4H/1H) e direção final.
    Retorna (label, color_tag)
    """
    htf = [r for r in results if r.tf in ("1D", "4H", "1H")]
    buy = sum(r.direction == "BUY" for r in htf)
    sell = sum(r.direction == "SELL" for r in htf)

    if buy >= 2:
        return "Alta (HTF)", "green"
    if sell >= 2:
        return "Baixa (HTF)", "red"
    return "Lateral (HTF)", "muted"

def classify_strength(results: List[TFResult]) -> Tuple[str, str]:
    """
    Força baseada em ADX médio ponderado dos TFs maiores.
    """
    # dá mais peso pra 1H/4H/1D
    pick = [r for r in results if r.tf in ("1D", "4H", "1H", "15M")]
    if not pick:
        pick = results

    weights = [TF_WEIGHTS.get(r.tf, 1.0) for r in pick]
    adx_val = float(np.average([r.adx14 for r in pick], weights=weights))

    if adx_val >= 28:
        return f"Forte (ADX {adx_val:.1f})", "green"
    if adx_val >= 20:
        return f"Média (ADX {adx_val:.1f})", "muted"
    return f"Fraca (ADX {adx_val:.1f})", "red"

def classify_volatility(tf_exec: TFResult) -> Tuple[str, str]:
    """
    Volatilidade baseada no ATR relativo ao preço no timeframe de execução (1M/5M).
    """
    rel = tf_exec.atr14 / max(tf_exec.close, 1e-9)
    # thresholds ajustados pra forex/metal
    if rel >= 0.0012:
        return f"Alta (ATR {rel*100:.2f}%)", "red"
    if rel >= 0.0006:
        return f"Média (ATR {rel*100:.2f}%)", "muted"
    return f"Baixa (ATR {rel*100:.2f}%)", "green"

def build_trade_levels(price: float, direction: str, atr_ref: float) -> Tuple[float, float, List[float]]:
    entry = price
    risk = max(atr_ref * 1.5, price * 0.0005)
    if direction == "BUY":
        stop = entry - risk
        tps = [entry + risk * i for i in range(1, 11)]
    elif direction == "SELL":
        stop = entry + risk
        tps = [entry - risk * i for i in range(1, 11)]
    else:
        stop = entry
        tps = []
    return entry, stop, tps

# ----------------------------
# UI Styling (Vortex-like)
# ----------------------------
st.markdown(
    """
    <style>
      .topbar { font-size: 26px; font-weight: 900; }
      .sub { color: rgba(255,255,255,0.65); margin-top: -6px; }
      .kpi {
        background: #12182a;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
      }
      .kpi h4 { margin: 0 0 6px 0; font-size: 14px; color: rgba(255,255,255,0.75); font-weight: 700; }
      .kpi .v { font-size: 18px; font-weight: 900; }
      .green { color: #00ff9c; }
      .red { color: #ff404d; }
      .muted { color: rgba(255,255,255,0.75); }
      .card {
        background: #12182a;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        margin-top: 12px;
      }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 900;
        border: 1px solid rgba(255,255,255,0.10);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="topbar">Vortex-Like Forex AI — Top-Down (1D → 1M)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Forex mercado aberto • Execução em M1/M5 • Painel completo de tendência/força/volatilidade/probabilidade</div>', unsafe_allow_html=True)

left, main = st.columns([1, 3], gap="large")

with left:
    st.subheader("Config")
    asset = st.radio("Ativo", list(ASSETS.keys()), index=1)
    exec_tf = st.selectbox("Timeframe de entrada", ["1M", "5M"], index=1)
    run = st.button("Analisar", use_container_width=True)
    auto = st.checkbox("Auto análise (a cada 5s)", value=False)
    st.caption("⚠️ Educacional. Não é recomendação financeira.")

with main:
    if auto:
        st.session_state["_tick"] = st.session_state.get("_tick", 0) + 1
        time.sleep(0.1)
        st.rerun()

    if "payload" not in st.session_state:
        st.session_state["payload"] = None

    if run or auto:
        symbol = ASSETS[asset]

        # 1H (base para 4H agregado)
        df_1h = fetch_ohlc(symbol, *TF_FETCH["1H"])
        df_4h = aggregate_to_4h_from_1h(df_1h) if not df_1h.empty else pd.DataFrame()

        # 1D
        df_1d = fetch_ohlc(symbol, *TF_FETCH["1D"])
        # 15M/5M/1M
        df_15m = fetch_ohlc(symbol, *TF_FETCH["15M"])
        df_5m = fetch_ohlc(symbol, *TF_FETCH["5M"])
        df_1m = fetch_ohlc(symbol, *TF_FETCH["1M"])

        # validações mínimas
        if df_1d.empty or df_1h.empty or df_15m.empty or df_5m.empty or df_1m.empty:
            st.error("Não consegui puxar todos os timeframes agora. Tente novamente em alguns segundos.")
            st.stop()

        results: List[TFResult] = []
        results.append(analyze_tf(df_1d, "1D"))

        # 4H: usa agregado; se tiver pouco dado, usa 1H como proxy
        if df_4h.empty or len(df_4h) < 60:
            results.append(analyze_tf(df_1h, "4H"))
        else:
            results.append(analyze_tf(df_4h, "4H"))

        results.append(analyze_tf(df_1h, "1H"))
        results.append(analyze_tf(df_15m, "15M"))

        res_5m = analyze_tf(df_5m, "5M")
        res_1m = analyze_tf(df_1m, "1M")
        results.append(res_5m)
        results.append(res_1m)

        final_dir, prob, norm, align_pct, conflict = combine_top_down(results)

        exec_res = res_1m if exec_tf == "1M" else res_5m

        # Regra de filtro: não operar contra o TF de execução
        status = "Aguardando"
        direction_to_trade = "NEUTRO"
        prob_adj = prob

        if final_dir == "NEUTRO":
            status = "Confluência fraca"
        else:
            if exec_res.direction != "NEUTRO" and exec_res.direction != final_dir:
                status = f"Bloqueado: {exec_tf} contra o Top-Down"
                direction_to_trade = "NEUTRO"
                prob_adj = max(55, prob - 12)
            else:
                status = f"Operação ativa ({final_dir})"
                direction_to_trade = final_dir

        # KPIs estilo Vortex
        trend_label, trend_color = classify_trend(results)
        strength_label, strength_color = classify_strength(results)
        vol_label, vol_color = classify_volatility(exec_res)

        price = exec_res.close
        atr_ref = res_5m.atr14 if not np.isnan(res_5m.atr14) else exec_res.atr14
        entry, stop, tps = build_trade_levels(price, direction_to_trade, atr_ref)

        table = pd.DataFrame([{
            "TF": r.tf,
            "Direção": r.direction,
            "Score": r.score,
            "Confiança%": r.confidence,
            "RSI": round(r.rsi14, 2),
            "ADX": round(r.adx14, 2),
            "Preço": round(r.close, 6),
        } for r in results])

        chart_df = (df_1m if exec_tf == "1M" else df_5m)[["Close"]].tail(300)

        st.session_state["payload"] = {
            "asset": asset,
            "price": price,
            "final_dir": direction_to_trade,
            "prob": int(prob_adj),
            "status": status,
            "trend": (trend_label, trend_color),
            "strength": (strength_label, strength_color),
            "vol": (vol_label, vol_color),
            "alignment": align_pct,
            "conflict": conflict,
            "entry": entry,
            "stop": stop,
            "tps": tps,
            "table": table,
            "chart_df": chart_df,
        }

    payload = st.session_state.get("payload")

    if payload is None:
        st.info("Clique em **Analisar** para gerar o painel completo.")
    else:
        # -------- KPIs --------
        k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

        def kpi_box(col, title, value, color="muted"):
            col.markdown(
                f"""
                <div class="kpi">
                  <h4>{title}</h4>
                  <div class="v {color}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        trend_label, trend_color = payload["trend"]
        strength_label, strength_color = payload["strength"]
        vol_label, vol_color = payload["vol"]

        kpi_box(k1, "Ativo", payload["asset"], "muted")
        kpi_box(k2, "Tendência", trend_label, trend_color)
        kpi_box(k3, "Força", strength_label, strength_color)
        kpi_box(k4, "Volatilidade", vol_label, vol_color)

        if payload["final_dir"] == "BUY":
            sig = "BUY"
            sig_color = "green"
        elif payload["final_dir"] == "SELL":
            sig = "SELL"
            sig_color = "red"
        else:
            sig = "NEUTRO"
            sig_color = "muted"

        kpi_box(k5, "Probabilidade", f'{payload["prob"]}% • {sig}', sig_color)

        # -------- Main cards --------
        st.markdown(
            f"""
            <div class="card">
              <div><b>Preço (execução {("M1" if payload["chart_df"].shape[0] > 0 else "")}):</b> {payload["price"]:.6f}</div>
              <div style="margin-top:6px;">
                <span class="pill">Status: {payload["status"]}</span>
                <span class="pill">Alinhamento: {payload["alignment"]}%</span>
                <span class="pill">Conflito: {payload["conflict"]}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
              <div><b>Entrada:</b> {payload["entry"]:.6f}</div>
              <div><b>Stop:</b> {payload["stop"]:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card"><b>Take Profits (1R → 10R)</b>', unsafe_allow_html=True)
        if payload["tps"]:
            tp_text = "\n".join([f"TP{i}: {tp:.6f}" for i, tp in enumerate(payload["tps"], start=1)])
            st.code(tp_text)
        else:
            st.write("Sem TPs (NEUTRO ou bloqueado por conflito).")
        st.markdown("</div>", unsafe_allow_html=True)

        # -------- Table + Chart --------
        cA, cB = st.columns([2, 2], gap="large")

        with cA:
            st.subheader("Top-Down por Timeframe")
            st.dataframe(payload["table"], use_container_width=True, hide_index=True)

        with cB:
            st.subheader(f"Gráfico (fechamento — {payload['asset']} — execução {exec_tf})")
            st.line_chart(payload["chart_df"], height=320)
