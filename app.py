import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# FIREBASE (CORRIGIDO)
# =========================
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Shark Black Institutional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🔥 TESTE FIREBASE (NOVO)
# =========================
st.sidebar.markdown("## 🔐 Firebase")

if st.sidebar.button("Ver usuários"):
    try:
        users = db.collection("users").stream()
        lista = []
        for user in users:
            lista.append(user.to_dict())

        st.sidebar.write(lista)
    except Exception as e:
        st.sidebar.error(f"Erro Firebase: {e}")

# =========================
# ESTILO (SEU ORIGINAL)
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #050505 0%, #0a0a0a 50%, #111111 100%);
        color: #f5f5f5;
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

REFRESH_SECONDS = 60

# =========================
# API
# =========================
def get_api_key():
    return st.secrets.get("TWELVE_DATA_API_KEY", None)

@st.cache_data(ttl=20)
def fetch_data(symbol, interval, apikey):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": apikey,
    }

    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    return df

# =========================
# UI SIMPLES (TESTE)
# =========================
st.title("SHARK BLACK")

api_key = get_api_key()

if not api_key:
    st.error("API KEY não encontrada")
else:
    asset = st.selectbox("Ativo", list(ASSETS.keys()))

    df = fetch_data(ASSETS[asset], "5min", api_key)

    if not df.empty:
        st.line_chart(df["Close"])
    else:
        st.warning("Sem dados")
