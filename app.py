import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# 주식 데이터 로드
@st.cache_data
def load_stock_data(ticker):
    data = yf.download(ticker, period="10y")
    data.reset_index(inplace=True)
    return data.copy()

# 모델 학습 함수
@st.cache_resource
def train_model(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    features = ['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume']
    target = 'Close'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=5)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    return model, rmse

# 주식 종가 예측 함수
def predict_stock_prices(model, ticker):
    future_dates = pd.date_range(start="2024-06-01", end="2024-12-30", freq='B')  # Business days
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day
    })
    
    last_close = yf.download(ticker, period="1d")['Close'][-1]
    future_data['Open'] = last_close
    future_data['High'] = last_close
    future_data['Low'] = last_close
    future_data['Volume'] = 1000000  # Dummy volume
    
    future_data.set_index('Date', inplace=True)
    predictions = model.predict(future_data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume']])
    future_data['Predicted Close'] = predictions
    
    return future_data

# Streamlit 애플리케이션
st.title("주가 예측 애플리케이션")

# 주식 티커 입력
ticker = st.text_input("주식 티커를 입력하세요", "AAPL")

# 데이터 로드 및 모델 학습
if st.button("데이터 로드 및 모델 학습"):
    with st.spinner("데이터를 로드 중입니다..."):
        stock_data = load_stock_data(ticker)
        st.write(f"{ticker} 주식 데이터")
        st.dataframe(stock_data.tail())
        
        with st.spinner("모델을 학습 중입니다... 잠시만 기다려주세요."):
            model, rmse = train_model(stock_data)
            st.session_state.model = model  # 모델을 세션 상태에 저장
            st.session_state.rmse = rmse
            st.success(f"모델 학습 완료! RMSE: {rmse:.2f}")

# 주식 종가 예측 및 그래프 표시
if st.button("2024년 주식 종가 예측"):
    if 'model' in st.session_state:
        with st.spinner("2024년 주식 종가를 예측 중입니다..."):
            future_data = predict_stock_prices(st.session_state.model, ticker)
            st.write(f"{ticker}  2024 Stock price prediction")
            st.dataframe(future_data)
            
            plt.figure(figsize=(10, 6))
            plt.plot(future_data.index, future_data['Predicted Close'], label='Predicted Close')
            plt.title(f"{ticker} 2024 Stock price prediction")
            plt.xlabel('Date')
            plt.ylabel('Predicted Close Price')
            plt.legend()
            st.pyplot(plt)
    else:
        st.error("먼저 모델을 학습시켜주세요.")