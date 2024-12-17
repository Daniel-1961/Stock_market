import pandas as pd
import numpy as np
import talib 
import matplotlib.pyplot as plt
import seaborn as sns
import pynance as pn

df=pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\AAPL_historical_data.csv")
#ata type and null value checked for this datatype

print(df.dtypes)
print(df.isnull().sum())
print(df.head())

df['Date']=pd.to_datetime(df['Date'],format='mixed', errors='coerce',utc=True)
print(df['Date'])


#Calculate Simple Moving Average (SMA)

def moving_average():
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)  # 20-day moving average

    # Calculate Exponential Moving Average (EMA)

    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['SMA_20'], label='SMA (20)', color='green')
    plt.plot(df['EMA_20'], label='EMA (20)', color='red')
    plt.title('Moving Averages')
    plt.legend()
    plt.grid()
    plt.show()

moving_average()


Calculate RSI
def relative_index():

    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # Plot RSI
    plt.figure(figsize=(12, 6))
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    plt.grid()
    plt.show()

relative_index()

#Calculate MACD

def average_convergence_diveregence():

    df['MACD'], df['Signal'], df['Hist'] = talib.MACD(df['Close'],fastperiod=12,slowperiod=26,signalperiod=9)
                 

    # Plot MACD
    plt.figure(figsize=(12, 6))
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['Signal'], label='Signal Line', color='orange')
    plt.bar(df.index, df['Hist'], label='Histogram', color='grey')
    plt.title('MACD Indicator')
    plt.legend()
    plt.grid()
    plt.show()

average_convergence_diveregence()

#calculating finanicial matrixs using pynance 

def finanicial_matric():

    df['Daily_return']=df['Close'].pct_change()
    print(df[['Date','Close','Daily_return']].head())
    df['Rolling_mean']=df['Close'].rolling(window=20).mean()
    df['rolling_Std']=df['Close'].rolling(window=20).std()
    print(df[['Date','Rolling_mean', 'rolling_Std']].head(100))

    #finanicial matrix visiualization

    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Daily_return'], label='Daily Return', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid()
    plt.legend()
    plt.show()






