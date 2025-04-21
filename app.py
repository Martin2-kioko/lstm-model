import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
data = pd.read_csv("MVS.csv", parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Load models & scalers
model_m = load_model("lstm_model_mastercard.h5", compile=False)
model_v = load_model("lstm_model_visa.h5", compile=False)

scaler_m = joblib.load("scaler_mastercard.pkl")
scaler_v = joblib.load("scaler_visa.pkl")
scaler_detrended_m = joblib.load("scaler_detrended_mastercard.pkl")
scaler_detrended_v = joblib.load("scaler_detrended_visa.pkl")
trend_model_m = joblib.load("trend_model_mastercard.pkl")
trend_model_v = joblib.load("trend_model_visa.pkl")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["🏠 Home", "📈 Predict Prices", "📊 Stock Info"])

if page == "🏠 Home":
    st.title("📊 Historical Trends: Visa & Mastercard")

    st.subheader("Line Chart of Stock Prices (2008 - 2024)")
    fig1 = px.line(data, y=['Close_M', 'Close_V'], labels={'value': 'Price', 'Date': 'Date'}, title="Mastercard vs Visa Prices")
    st.plotly_chart(fig1)

    st.subheader("Volume Traded Pie Chart")
    total_volume = {
        'Mastercard': data['Volume_M'].sum(),
        'Visa': data['Volume_V'].sum()
    }
    fig2 = px.pie(names=total_volume.keys(), values=total_volume.values(), title="Total Volume Traded")
    st.plotly_chart(fig2)

elif page == "📈 Predict Prices":
    st.title("📈 Stock Price Prediction (Mastercard & Visa)")
    option = st.selectbox("Select Stock", ("Mastercard", "Visa"))

    future_date = st.date_input("Enter a future date beyond 2024", value=datetime(2025, 1, 1))
    seq_length = 60

    # Select relevant data
    if option == "Mastercard":
        stock_data = data[['Close_M', 'Volume_M']]
        close_col = 'Close_M'
        volume_col = 'Volume_M'
        scaler = scaler_m
        scaler_detrended = scaler_detrended_m
        trend_model = trend_model_m
        model = model_m
    else:
        stock_data = data[['Close_V', 'Volume_V']]
        close_col = 'Close_V'
        volume_col = 'Volume_V'
        scaler = scaler_v
        scaler_detrended = scaler_detrended_v
        trend_model = trend_model_v
        model = model_v

    # Compute features
    stock_data['Returns'] = stock_data[close_col].pct_change()
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std().bfill()
    stock_data['MA_Close'] = stock_data[close_col].rolling(window=20).mean().bfill()
    stock_data['SMA50'] = stock_data[close_col].rolling(window=50).mean().bfill()
    stock_data['SMA200'] = stock_data[close_col].rolling(window=200).mean().bfill()

    features = stock_data[[close_col, volume_col, 'Volatility', 'MA_Close', 'SMA50', 'SMA200', 'Returns']].dropna()
    features['Volatility'] = features['Volatility'].rolling(window=5).mean().bfill()

    # Detrend
    X_idx = np.arange(len(features)).reshape(-1, 1)
    detrended = features[close_col].values - trend_model.predict(X_idx)

    # 🔧 FIX: Convert to numpy array to avoid feature name errors
    scaled_features = scaler.transform(features.values)
    scaled_detrended = scaler_detrended.transform(detrended.reshape(-1, 1))

    # Sequence
    def create_sequences(features, target, seq_length):
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:i + seq_length])
            y.append(target[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_features, scaled_detrended, seq_length)
    predicted_detrended = model.predict(X)
    predicted_detrended = scaler_detrended.inverse_transform(predicted_detrended)

    # Add trend back
    future_X = np.arange(len(features) - len(predicted_detrended), len(features)).reshape(-1, 1)
    trend = trend_model.predict(future_X)
    predicted_prices = predicted_detrended.flatten() + trend
    actual_prices = features[close_col].iloc[seq_length:]

    # Filter by future date
    filtered_index = actual_prices.index[actual_prices.index > pd.to_datetime("2024-12-31")]

    st.subheader(f"{option} Predicted Prices After 2024")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_prices.index[-len(predicted_prices):], predicted_prices, label='Predicted', color='green')
    ax.set_title(f"{option} Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

elif page == "📊 Stock Info":
    st.title("📘 About Mastercard & Visa Stock Markets")

    st.markdown("""
    ### Mastercard Inc. (MA)
    Mastercard is a global payments technology company that connects consumers, financial institutions, merchants, governments, and businesses worldwide. The company's primary revenue comes from fees paid by financial institutions.

    - **Founded**: 1966  
    - **Ticker**: MA (NYSE)  
    - **Sector**: Financial Services

    ### Visa Inc. (V)
    Visa operates the world's largest retail electronic payments network and is one of the most recognized global financial services brands. It facilitates digital fund transfers worldwide.

    - **Founded**: 1958  
    - **Ticker**: V (NYSE)  
    - **Sector**: Financial Services

    ### Market Summary
    Both Mastercard and Visa have demonstrated strong growth and resilience in the financial technology space. Their business models emphasize transaction volume, not credit risk, making them relatively stable investments.
    """)
