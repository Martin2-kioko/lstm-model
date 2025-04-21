import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load data
data = pd.read_csv("MVS.csv", parse_dates=['Date'])
data.set_index('Date', inplace=True)
data['Year'] = data.index.year

# Load models & scalers
model_m = load_model("lstm_model_mastercard.h5", compile=False)
model_v = load_model("lstm_model_visa.h5", compile=False)

scaler_m = joblib.load("scaler_mastercard.pkl")
scaler_v = joblib.load("scaler_visa.pkl")
scaler_detrended_m = joblib.load("scaler_detrended_mastercard.pkl")
scaler_detrended_v = joblib.load("scaler_detrended_visa.pkl")
trend_model_m = joblib.load("trend_model_mastercard.pkl")
trend_model_v = joblib.load("trend_model_visa.pkl")

st.set_page_config(page_title="Visa & Mastercard Stock Prediction", layout="wide")
st.sidebar.title("📊 Stock Dashboard")

page = st.sidebar.radio("Navigation", ["🏠 Home", "📈 Predictions", "📘 Stock Info"])

if page == "🏠 Home":
    st.title("📊 Historical Trends: Visa & Mastercard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock Prices Over Time")
        fig1 = px.line(data, y=['Close_M', 'Close_V'], labels={'value': 'Price', 'Date': 'Date'},
                      title="Mastercard vs Visa Prices")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Yearly Volume Traded")
        yearly_volume = data.groupby('Year')[['Volume_M', 'Volume_V']].sum().reset_index()
        fig2 = px.bar(yearly_volume, x='Year', y=['Volume_M', 'Volume_V'],
                     labels={'value': 'Volume', 'variable': 'Company'}, barmode='group',
                     title="Yearly Volume Traded for Mastercard & Visa")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "📈 Predictions":
    st.title("📈 Predict Future Stock Prices")
    future_date = st.date_input("Enter a future date (post-2024)", value=datetime(2025, 1, 1))
    seq_length = 60

    def prepare_prediction_data(option):
        if option == "Mastercard":
            stock_data = data[["Close_M", "Volume_M"]].copy()
            close_col = 'Close_M'
            volume_col = 'Volume_M'
            scaler = scaler_m
            scaler_detrended = scaler_detrended_m
            trend_model = trend_model_m
            model = model_m
        else:
            stock_data = data[["Close_V", "Volume_V"]].copy()
            close_col = 'Close_V'
            volume_col = 'Volume_V'
            scaler = scaler_v
            scaler_detrended = scaler_detrended_v
            trend_model = trend_model_v
            model = model_v

        stock_data['Returns'] = stock_data[close_col].pct_change()
        stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std().bfill()
        stock_data['MA_Close'] = stock_data[close_col].rolling(window=20).mean().bfill()
        stock_data['SMA50'] = stock_data[close_col].rolling(window=50).mean().bfill()
        stock_data['SMA200'] = stock_data[close_col].rolling(window=200).mean().bfill()

        features = stock_data[[close_col, volume_col, 'Volatility', 'MA_Close', 'SMA50', 'SMA200', 'Returns']].dropna()
        features['Volatility'] = features['Volatility'].rolling(window=5).mean().bfill()

        X_idx = np.arange(len(features)).reshape(-1, 1)
        detrended = features[close_col].values - trend_model.predict(X_idx)

        scaled_features = scaler.transform(features.values)
        scaled_detrended = scaler_detrended.transform(detrended.reshape(-1, 1))

        def create_sequences(features, target, seq_length):
            X, y = [], []
            for i in range(len(features) - seq_length):
                X.append(features[i:i + seq_length])
                y.append(target[i + seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_features, scaled_detrended, seq_length)
        predicted_detrended = model.predict(X)
        predicted_detrended = scaler_detrended.inverse_transform(predicted_detrended)

        future_X = np.arange(len(features) - len(predicted_detrended), len(features)).reshape(-1, 1)
        trend = trend_model.predict(future_X)
        predicted_prices = predicted_detrended.flatten() + trend
        actual_prices = features[close_col].iloc[seq_length:]

        return actual_prices, predicted_prices

    st.subheader("Compare Mastercard and Visa")
    actual_m, predicted_m = prepare_prediction_data("Mastercard")
    actual_v, predicted_v = prepare_prediction_data("Visa")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_m.index[-len(predicted_m):], y=actual_m.values[-len(predicted_m):],
                             mode='lines', name='Mastercard Actual'))
    fig.add_trace(go.Scatter(x=actual_m.index[-len(predicted_m):], y=predicted_m,
                             mode='lines', name='Mastercard Predicted'))
    fig.add_trace(go.Scatter(x=actual_v.index[-len(predicted_v):], y=actual_v.values[-len(predicted_v):],
                             mode='lines', name='Visa Actual'))
    fig.add_trace(go.Scatter(x=actual_v.index[-len(predicted_v):], y=predicted_v,
                             mode='lines', name='Visa Predicted'))
    fig.update_layout(title="Mastercard vs Visa: Historical & Predicted Prices",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Suggestion
    if predicted_m[-1] > actual_m.values[-1] and predicted_v[-1] > actual_v.values[-1]:
        suggestion = "Both Mastercard and Visa show an upward trend. Consider investing in either, depending on diversification needs."
    elif predicted_m[-1] > actual_m.values[-1]:
        suggestion = "Mastercard shows a strong upward trend. It may be a good investment opportunity."
    elif predicted_v[-1] > actual_v.values[-1]:
        suggestion = "Visa is trending upward. It may be worth considering for investment."
    else:
        suggestion = "Both stocks show a downward or flat trend. Monitor further before investing."

    st.success(f"💡 Investment Suggestion: {suggestion}")

elif page == "📘 Stock Info":
    st.title("📘 About Mastercard & Visa Stock Markets")
    st.markdown("""
    ### Mastercard Inc. (MA)
    Mastercard is a global payments technology company connecting consumers, businesses, and financial institutions worldwide. It earns revenue primarily from transaction fees.

    - **Founded**: 1966  
    - **Ticker**: MA (NYSE)  
    - **Sector**: Financial Services

    ### Visa Inc. (V)
    Visa operates the largest retail electronic payments network and is a globally recognized financial services brand.

    - **Founded**: 1958  
    - **Ticker**: V (NYSE)  
    - **Sector**: Financial Services

    ### Summary
    Mastercard and Visa offer relatively stable investment opportunities due to their low exposure to credit risk and reliance on global transaction volume.
    """)
