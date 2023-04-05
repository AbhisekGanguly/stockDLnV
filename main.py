import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# Importing the prediction library
from prediction import predict_stock

# Import the technical analysis library
import talib as ta

# Importing alpha vantage fundamentals
from alpha_vantage.fundamentaldata import FundamentalData

# Importing stocknews API
from stocknews import StockNews

# Make an interactive stock dashboard
# Add a title
st.title('Stock Dashboard')

# Add a sidebar with ticker, start date and end date
ticker = st.sidebar.text_input('Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date')

# Get the data using yfinance and store it to a variable for further use
data = yf.download(ticker, start_date, end_date)
data_copy = data.copy()

# Add the plotly chart
fig  = px.line(data, x = data.index, y = data['Adj Close'], title=ticker.upper())
# Center the title
fig.update_layout(title_x=0.5)

# Add different stock indicators options to the chart
indicators = st.sidebar.multiselect('Indicators', options=['SMA', 'EMA', 'BBANDS', 'MACD', 'ADX', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH'])

# Add indicators columns to the data
data['SMA'] = data['Adj Close'].rolling(window=20).mean()
data['EMA'] = data['Adj Close'].ewm(span=20, adjust=False).mean()
data['BBANDS_upperband'], data['BBANDS_middleband'], data['BBANDS_lowerband'] = ta.BBANDS(data['Adj Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['ADX'] = ta.ADX(data['High'], data['Low'], data['Adj Close'], timeperiod=14)
data['AROONOSC'] = ta.AROONOSC(data['High'], data['Low'], timeperiod=14)
data['CCI'] = ta.CCI(data['High'], data['Low'], data['Adj Close'], timeperiod=14)
data['CMO'] = ta.CMO(data['Adj Close'], timeperiod=14)
data['DX'] = ta.DX(data['High'], data['Low'], data['Adj Close'], timeperiod=14)
data['MFI'] = ta.MFI(data['High'], data['Low'], data['Adj Close'], data['Volume'], timeperiod=14)
data['MINUS_DI'] = ta.MINUS_DI(data['High'], data['Low'], data['Adj Close'], timeperiod=14)
data['MINUS_DM'] = ta.MINUS_DM(data['High'], data['Low'], timeperiod=14)
data['MOM'] = ta.MOM(data['Adj Close'], timeperiod=10)
data['PLUS_DI'] = ta.PLUS_DI(data['High'], data['Low'], data['Adj Close'], timeperiod=14)
data['PLUS_DM'] = ta.PLUS_DM(data['High'], data['Low'], timeperiod=14)
data['PPO'] = ta.PPO(data['Adj Close'], fastperiod=12, slowperiod=26, matype=0)
data['ROC'] = ta.ROC(data['Adj Close'], timeperiod=10)
data['ROCP'] = ta.ROCP(data['Adj Close'], timeperiod=10)
data['ROCR'] = ta.ROCR(data['Adj Close'], timeperiod=10)
data['ROCR100'] = ta.ROCR100(data['Adj Close'], timeperiod=10)
data['RSI'] = ta.RSI(data['Adj Close'], timeperiod=14)
data['STOCH_slowk'], data['STOCH_slowd'] = ta.STOCH(data['High'], data['Low'], data['Adj Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
data['STOCHF_fastk'], data['STOCHF_fastd'] = ta.STOCHF(data['High'], data['Low'], data['Adj Close'], fastk_period=5, fastd_period=3, fastd_matype=0)


# Add the indicators to the chart
for indicator in indicators:
    if indicator == 'SMA':
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', line=dict(color='blue'), name='SMA'))
    if indicator == 'EMA':
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], mode='lines', line=dict(color='red'), name='EMA'))
    if indicator == 'BBANDS':
        fig.add_trace(go.Scatter(x=data.index, y=data['BBANDS_upperband'], mode='lines', line=dict(color='green'), name='Upper Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BBANDS_middleband'], mode='lines', line=dict(color='gray'), name='Middle Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BBANDS_lowerband'], mode='lines', line=dict(color='red'), name='Lower Band'))
    if indicator == 'MACD':
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', line=dict(color='blue'), name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', line=dict(color='red', dash='dot'), name='MACD Signal'))
    if indicator == 'ADX':
        fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', line=dict(color='yellow'), name='ADX'))
    if indicator == 'AROONOSC':
        fig.add_trace(go.Scatter(x=data.index, y=data['AROONOSC'], mode='lines', line=dict(color='blue'), name='AROONOSC'))
    if indicator == 'CCI':
        fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', line=dict(color='green'), name='CCI'))
    if indicator == 'CMO':
        fig.add_trace(go.Scatter(x=data.index, y=data['CMO'], mode='lines', line=dict(color='gray'), name='CMO'))
    if indicator == 'DX':
        fig.add_trace(go.Scatter(x=data.index, y=data['DX'], mode='lines', line=dict(color='aqua'), name='DX'))
    if indicator == 'MFI':
        fig.add_trace(go.Scatter(x=data.index, y=data['MFI'], mode='lines', line=dict(color='magenta'), name='MFI'))
    if indicator == 'MINUS_DI':
        fig.add_trace(go.Scatter(x=data.index, y=data['MINUS_DI'], mode='lines', line=dict(color='red'), name='MINUS_DI'))
    if indicator == 'MINUS_DM':
        fig.add_trace(go.Scatter(x=data.index, y=data['MINUS_DM'], mode='lines', line=dict(color='pink'), name='MINUS_DM'))
    if indicator == 'MOM':
        fig.add_trace(go.Scatter(x=data.index, y=data['MOM'], mode='lines', line=dict(color='green'), name='MOM'))
    if indicator == 'PLUS_DI':
        fig.add_trace(go.Scatter(x=data.index, y=data['PLUS_DI'], mode='lines', line=dict(color='orange'), name='PLUS_DI'))
    if indicator == 'PLUS_DM':
        fig.add_trace(go.Scatter(x=data.index, y=data['PLUS_DM'], mode='lines', line=dict(color='yellow'), name='PLUS_DM'))
    if indicator == 'PPO':
        fig.add_trace(go.Scatter(x=data.index, y=data['PPO'], mode='lines', line=dict(color='blue'), name='PPO'))
    if indicator == 'ROC':
        fig.add_trace(go.Scatter(x=data.index, y=data['ROC'], mode='lines', line=dict(color='pink'), name='ROC'))
    if indicator == 'ROCP':
        fig.add_trace(go.Scatter(x=data.index, y=data['ROCP'], mode='lines', line=dict(color='green'), name='ROCP'))
    if indicator == 'ROCR':
        fig.add_trace(go.Scatter(x=data.index, y=data['ROCR'], mode='lines', line=dict(color='red'), name='ROCR'))
    if indicator == 'ROCR100':
        fig.add_trace(go.Scatter(x=data.index, y=data['ROCR100'], mode='lines', line=dict(color='blue'), name='ROCR100'))
    if indicator == 'RSI':
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', line=dict(color='gray'), name='RSI'))
    if indicator == 'STOCH':
        fig.add_trace(go.Scatter(x=data.index, y=data['STOCH_slowk'], mode='lines', line=dict(color='blue'), name='STOCH_slowk'))
        fig.add_trace(go.Scatter(x=data.index, y=data['STOCH_slowd'], mode='lines', line=dict(color='red'), name='STOCH_slowd'))
st.plotly_chart(fig)

# Add tab options to choose from pricing data, Fundamental data and top 10 news 
pricing_data, fundamental_data, news_data, predictions = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "Predictions"])

stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)

# Adding information to the tabs
with pricing_data:
    st.header("Price Movement")    
    st.write(stock_data)
    # print annual returns, standard deviation and risk adjusted return
    st.write("Annual Returns: ", round(stock_data['Close'].pct_change().mean() * 252, 2), "%")
    st.write("Standard Deviation: ", round(stock_data['Close'].pct_change().std() * np.sqrt(252), 2))
    st.write("Risk Adjusted Return: ", round(stock_data['Close'].pct_change().mean() * 252 / stock_data['Close'].pct_change().std() * np.sqrt(252), 2), "%")

# Using alpha vantage to get fundamental data
with fundamental_data:
    st.header("Fundamental Data")

    key = 'SFD3S4WWG0P8OCGI'
    fd = FundamentalData(key, output_format='pandas')
    # Balance Sheet
    st.subheader("Balance Sheet")
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)

    # Income statement
    st.subheader("Income Statement")
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is_ = income_statement.T[2:]
    is_.columns = list(income_statement.T.iloc[0])
    st.write(is_)

    # Cashflow Statement
    st.subheader("Cashflow Statement")
    cashflow_statement = fd.get_cash_flow_annual(ticker)[0]
    cf = cashflow_statement.T[2:]
    cf.columns = list(cashflow_statement.T.iloc[0])
    st.write(cf)

with news_data:
    st.header(f"Top 10 News of {ticker}")
    # Using stocknews API to get top 10 news
    sn = StockNews(ticker, save_news=False)
    news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News: {i+1}')
        st.write(news['published'][i])
        st.write(news['title'][i])
        st.write(news['summary'][i])
        title_sentiment = news['sentiment_title'][i]
        st.write(f'Title Sentiment: {title_sentiment}')
        summary_sentiment = news['sentiment_summary'][i]
        st.write(f'Summary Sentiment: {summary_sentiment}')

# Using prophet to predict stock price
with predictions:
    st.header("Next 30 Days Predictions")
    predict = predict_stock(ticker)
    st.write(predict)
