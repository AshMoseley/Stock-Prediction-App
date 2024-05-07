import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import math
from tensorflow.keras.callbacks import EarlyStopping
from plotly import graph_objs as go
import pandas as pd
import datetime as dt
from datetime import date
import  plotly.express as px

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "GME", "NVDA", "AMD"]


# ---------- Caching Data ----------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)  
    return data


# ---------- Displaying Financials ----------------------------
def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    info = df_ticker.info

    financial_data = {
        "Company Name": info.get('longName'),
        "Sector": info.get('sector'),
        "Website": info.get('website'),
        "Average Volume": f"{info.get('averageVolume', 0):,}", 
        "Market Cap": f"${info.get('marketCap', 0):,}",
        "Previous Close": f"${info.get('previousClose', 0):.2f}",
        "52 Week Change": f"{info.get('52WeekChange', 0) * 100:.2f}%",
        "52 Week High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
        "52 Week Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
        "200 Day Average": f"${info.get('twoHundredDayAverage', 0):.2f}"
    }
    financial_df = pd.DataFrame(list(financial_data.items()), columns=['Metric', 'Value'])
    st.table(financial_df.set_index('Metric'))

# ---------- Plotting Raw Data ----------------------------
def plot_raw_data(stock, data):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName'] 

    if 'Date' not in data.columns:
        data.reset_index(inplace=True)

    numeric_df = data.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()

    st.markdown('**Select Features to Plot**')
    features_selected = st.multiselect("Choose features", numeric_cols)

    if features_selected:
        cust_data = data[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data['Date'], y=features_selected,
                                title=f'{Name} - Selected Features Over Time')
        
        plotly_figure.update_layout(
            title={'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Variables',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(plotly_figure, use_container_width=True)
    else:
        st.warning("Please select at least one feature to plot.")


# ----------------- Model Training and Evaluation ----------------------

def evaluate_model_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2



# ------------- Data preparation for LSTM Model ----------------------
def create_train_test_data(df, history_size=60):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    x_train, y_train = [], []
    for i in range(history_size, len(features)):
        x_train.append(scaled_features[i-history_size:i])
        y_train.append(scaled_features[i, features.columns.get_loc('Close')])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshape for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features.columns)))
    return x_train, y_train, scaler



# ----------- Creating Training and Testing Data for LSTM Model ----------------
def train_LSTM_model(x_train, y_train, epochs, batch_size):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
              validation_split=0.1, callbacks=[early_stop], verbose=1)
    return model

# -------------- Finding Moving Average ---------------------------------------
def find_moving_avg(ma_days, data):
    if 'Date' not in data.columns:
        data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    data.loc[:, 'SMA'] = data['Close'].rolling(window=ma_days, min_periods=1).mean()

    data.reset_index(inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], mode='lines', name=f'{ma_days}-Day SMA'))

    fig.update_layout(
        title=f'{ma_days}-Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)



#------------ Plotting the Predictions -------------------------
def prediction_plot(model, df, scaler, history_size=60):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    dates = df['Date'].values 

    # Apply transformation using DataFrame to retain column names
    scaled_features = scaler.transform(features)

    # Prepare the data for LSTM input
    X_test = []
    for i in range(history_size, len(features)):
        X_test.append(scaled_features[i-history_size:i])
    X_test = np.array(X_test)

    # Get model predictions
    predicted_prices = model.predict(X_test)

    # Inverse transform the predictions
    dummy_for_inverse = pd.DataFrame(np.zeros((len(predicted_prices), scaled_features.shape[1])),
                                     columns=features.columns)
    dummy_for_inverse['Close'] = predicted_prices.flatten()  

    inverse_transformed = scaler.inverse_transform(dummy_for_inverse)
    predicted_close_prices = inverse_transformed[:, features.columns.get_loc('Close')]  

    actual_close_prices = features['Close'][history_size:].values

    mse, rmse, mae, r2 = evaluate_model_performance(actual_close_prices, predicted_close_prices)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[history_size:], y=actual_close_prices, mode='lines', name='Actual Close Price'))
    fig.add_trace(go.Scatter(x=dates[history_size:], y=predicted_close_prices, mode='lines', name='Predicted Close Price'))
    fig.update_layout(title='Actual vs Predicted Close Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.write(f"Model Evaluation Metrics:")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R^2 Score: {r2}")

# ---------------- Sidebar Menu -----------------------
st.sidebar.title("Settings")
activity = st.sidebar.radio("Select Activity", ["Explore Stocks", "Train Model"])

user_input = st.sidebar.text_input("Enter Stock Symbol", 'AAPL').upper()

if user_input:
    data = load_data(user_input)
    
    if activity == 'Explore Stocks':
        st.subheader(f"Stock Exploration for {user_input}")
        
        explore_option = st.radio("Choose to explore", ["Financials", "Price Data", "Moving Averages"])
        
        if explore_option == "Financials":
            stock_financials(user_input)
        elif explore_option == "Price Data":
            plot_raw_data(user_input, data)
        elif explore_option == "Moving Averages":
            ma_days = st.slider("Select Number of Days for Moving Average", 5, 200, 50)
            find_moving_avg(ma_days, data)

    elif activity == 'Train Model':
        st.subheader(f"Train Machine Learning Model for {user_input}")
        
        # Training configurations
        epochs = st.number_input("Set Epochs", min_value=10, max_value=300, value=50, step=10)
        batch_size = st.number_input("Set Batch Size", min_value=16, max_value=256, value=64)
        
        if st.button('Train Model'):
            # Training the model
            x_train, y_train, scaler = create_train_test_data(data, history_size=60)
            model = train_LSTM_model(x_train, y_train, epochs, batch_size)
            
            prediction_plot(model, data, scaler, history_size=60)
            st.success("Model trained successfully!")