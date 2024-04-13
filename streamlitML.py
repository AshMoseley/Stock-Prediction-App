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
TODAY = dt.datetime.now().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ["Select the Stock", "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "GME", "NVDA", "AMD"]


# Loading Data ---------------------

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)  
    return data


#For Stock Financials ----------------------

def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    marketCap = df_ticker.info['marketCap']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
    fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    ftWeekChange = df_ticker.info['52WeekChange']
    website = df_ticker.info['website']

    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Market Cap -', marketCap)
    st.write('Previous Close -', prevClose)
    st.write('52 Week Change -', ftWeekChange)
    st.write('52 Week High -', fiftyTwoWeekHigh)
    st.write('52 Week Low -', fiftyTwoWeekLow)
    st.write('200 Day Average -', twoHunDayAvg)


#Plotting Raw Data ---------------------------------------

def plot_raw_data(stock, data_1):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName']
    data_1.reset_index()
    numeric_df = data_1.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to **_Plot_**')
    features_selected = st.multiselect("", numeric_cols)
    if st.button("Generate Plot"):
        cust_data = data_1[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data_1['Date'], y=features_selected,
                                title= Name + ' ' + '<i>timeline</i>')
        plotly_figure.update_layout(title = {'y':0.9,'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
        plotly_figure.update_xaxes(title_text='Date')
        plotly_figure.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, title="Price"), width=800, height=550)
        st.plotly_chart(plotly_figure)



#For LSTM MOdel ------------------------------

# Data preparation
def create_train_test_data(df, history_size=60):
    # Assuming 'Date' is not used as an input feature for LSTM
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



#Creating Training and Testing Data for other Models ----------------------

# LSTM model with dropout and early stopping
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

#Finding Movinf Average ---------------------------------------

def find_moving_avg(ma_button, df):
    days = ma_button

    data1 = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data1)):
        new_data['Date'][i] = data1['Date'][i]
        new_data['Close'][i] = data1['Close'][i]

    new_data['SMA_'+str(days)] = new_data['Close'].rolling(min_periods=1, window=days).mean()

    #new_data.dropna(inplace=True)
    new_data.isna().sum()

    #st.write(new_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['SMA_'+str(days)], mode='lines', name='SMA_'+str(days)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)





#Plotting the Predictions -------------------------


# Prediction and plotting function
def prediction_plot(model, df, scaler, history_size=60):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    dates = df['Date'].values
    scaled_features = scaler.transform(features)

    X_test = []
    for i in range(history_size, len(features)):
        X_test.append(scaled_features[i-history_size:i])
    X_test = np.array(X_test)

    predicted_prices = model.predict(X_test)

    # Inverse transform for 'Close' predictions
    # Assuming 'Close' is the fourth column (index 3) in your features
    dummy_for_inverse = np.zeros((len(predicted_prices), scaled_features.shape[1]))
    dummy_for_inverse[:, 3] = predicted_prices.flatten()
    predicted_close_prices = scaler.inverse_transform(dummy_for_inverse)[:, 3]

    actual_close_prices = features[history_size:, 3]  # Index 3 for 'Close'

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[history_size:], y=actual_close_prices, mode='lines', name='Actual Close Price'))
    fig.add_trace(go.Scatter(x=dates[history_size:], y=predicted_close_prices, mode='lines', name='Predicted Close Price'))
    fig.update_layout(title='Actual vs Predicted Close Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)


    


 #Sidebar Menu -----------------------

menu=["Stock Exploration and Feature Extraction", "Train Model"]
st.sidebar.title("Settings")
st.sidebar.subheader("Timeseries Settings")
choices = st.sidebar.selectbox("Select the Activity", menu,index=0)



if choices == 'Stock Exploration and Feature Extraction':
    st.subheader("Extract Data")
    st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
    user_input = st.text_input("", '')

    if not user_input:
        pass
    else:
        data = load_data(user_input)
        st.markdown('Select from the options below to Explore Stocks')

        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration',
                                                     'Extract Features for Stock Price Forecasting'], index=0)
        if selected_explore == 'Stock Financials Exploration':
            st.markdown('')
            st.markdown('**_Stock_ _Financial_** Information ------')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_raw_data(user_input, data)
            st.markdown('')
            shw_SMA = st.checkbox('Show Moving Average')

            if shw_SMA:
                st.write('Stock Data based on Moving Average')
                st.write('A Moving Average(MA) is a stock indicator that is commonly used in technical analysis')
                st.write(
                    'The reason for calculating moving average of a stock is to help smooth out the price of data over '
                    'a specified period of time by creating a constanly updated average price')
                st.write(
                    'A Simple Moving Average (SMA) is a calculation that takes the arithmatic mean of a given set of '
                    'prices over the specified number of days in the past, for example: over the previous 15, 30, 50, '
                    '100, or 200 days.')

                ma_button = st.number_input("Select Number of Days Moving Average", 5, 200)

                if ma_button:
                    st.write('You entered the Moving Average for ', ma_button, 'days')
                    find_moving_avg(ma_button, data)

        elif selected_explore == 'Extract Features for Stock Price Forecasting':
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input("", date(2014, 1, 1))
            st.write('You Selected Data From - ', start_date)
            submit_button = st.button("Extract Features")

            start_row = 0
            if submit_button:
                st.write('Extracted Features Dataframe for ', user_input)
                for i in range(0, len(data)):
                    if start_date <= pd.to_datetime(data['Date'][i]):
                        start_row = i
                        break
                st.write(data.iloc[start_row:, :])

elif choices == 'Train Model':
    
    st.subheader("Train Machine Learning Models for Stock Prediction")
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    stock_select = st.selectbox("", stocks, index=0)
    
    if stock_select:
        df = load_data(stock_select)
        epoch = st.sidebar.slider("Epochs", min_value=10, max_value=300, value=100, step=10)
        b_s = st.sidebar.slider("Batch Size", min_value=16, max_value=1024, value=32, step=16)
        
        if st.sidebar.button('Train Model'):
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df[['Date','Close']])
            
            # Adjusted to correctly unpack the returned values from create_train_test_data
            x_train, y_train, scaler = create_train_test_data(df, history_size=60) # Make sure this matches your intended use
            
            # Since you're predicting 'Close' prices, ensure the input to create_train_test_data is correct
            model = train_LSTM_model(x_train, y_train, epochs=epoch, batch_size=b_s)
            
            prediction_plot(model, df, scaler, history_size=60)  # Assuming history_size=60 as before
            
            st.success("Model trained successfully!")
