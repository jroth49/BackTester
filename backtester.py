import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import traceback
import sys
import subprocess
import os

pd.options.mode.chained_assignment = None
intro_msg = r'''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠾⠛⢉⣉⣉⣉⡉⠛⠷⣦⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣠⣴⣿⣿⣿⣿⣿⡿⣿⣶⣌⠹⣷⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠁⣴⣿⣿⣿⣿⣿⣿⣿⣿⣆⠉⠻⣧⠘⣷⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠈⠀⢹⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⢸⣿⠛⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⢸⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣷⠀⢿⡆⠈⠛⠻⠟⠛⠉⠀⠀⠀⠀⠀⠀⣾⠃⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣧⡀⠻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠃⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢼⠿⣦⣄⠀⠀⠀⠀⠀⠀⠀⣀⣴⠟⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣦⠀⠀⠈⠉⠛⠓⠲⠶⠖⠚⠋⠉⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣠⣾⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣾⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢀⣄⠈⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
Progam Analysis
'''


#util functions
def read_historical_data(file_path):
    return pd.read_csv(file_path)

def SMA(ticker_df, win1, win2):

    ticker_df[f'SMA_{win1}'] = ticker_df['Close'].rolling(window=win1).mean()
    ticker_df[f'SMA_{win2}'] = ticker_df['Close'].rolling(window=win2).mean()
    
    return(ticker_df)

def RSI(ticker_df, period, buy_rsi, sell_rsi):

    ticker_df['Buy RSI'] = buy_rsi
    ticker_df['Sell RSI'] = sell_rsi
    ticker_df['Price Change'] = ticker_df['Close'].diff()
    ticker_df['Gain'] = ticker_df['Price Change'].where(ticker_df['Price Change'] > 0, 0)
    ticker_df['Loss'] = -ticker_df['Price Change'].where(ticker_df['Price Change'] < 0, 0)

    ticker_df['Avg Gain'] = ticker_df['Gain'].rolling(window=period).mean()
    ticker_df['Avg Loss'] = ticker_df['Loss'].rolling(window=period).mean()
    ticker_df['RS'] = ticker_df['Avg Gain'] / ticker_df['Avg Loss']
    ticker_df['RSI'] = 100 - (100 / (1 + ticker_df['RS']))

    return ticker_df

def BB(ticker_df, period, multiplier):

    ticker_df['Middle Band'] = ticker_df['Close'].rolling(window=period).mean()
    ticker_df['Standard Deviation'] = ticker_df['Close'].rolling(window=period).std()
    
    ticker_df['Upper Band'] = ticker_df['Middle Band'] + (ticker_df['Standard Deviation'] * (multiplier + 0.25))
    ticker_df['Lower Band'] = ticker_df['Middle Band'] - (ticker_df['Standard Deviation'] * multiplier)

    return ticker_df

def EMA(ticker_df, win1, win2):

    ticker_df[f'EMA_{win1}'] = ticker_df['Close'].ewm(span=win1, adjust=False).mean()
    ticker_df[f'EMA_{win2}'] = ticker_df['Close'].ewm(span=win2, adjust=False).mean()
    
    return ticker_df

def MACD(ticker_df):

    ticker_df['MACD'] = ticker_df['Close'].ewm(span=12, adjust=False).mean() - ticker_df['Close'].ewm(span=36, adjust=False).mean()
    ticker_df['MACD Line'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
    
    return ticker_df

def CMF(ticker_df, win):
    ticker_df['MFM'] = ((ticker_df['Close'] - ticker_df['Low']) - (ticker_df['High'] - ticker_df['Close'])) / (ticker_df['High'] - ticker_df['Low'])
    ticker_df['MFV'] = ticker_df['MFM'] * ticker_df['Volume']
    ticker_df['CMF'] = ticker_df['MFV'].rolling(window=win).sum() / ticker_df['Volume'].rolling(window=win).sum()

    return ticker_df

def STO(ticker_df, win):
    ticker_df['Low_min'] = ticker_df['Low'].rolling(win).min()  # lowest low over last n periods
    ticker_df['High_max'] = ticker_df['High'].rolling(win).max()  # highest high over last n periods

    ticker_df['%K'] = (ticker_df['Close'] - ticker_df['Low_min']) / (ticker_df['High_max'] - ticker_df['Low_min']) * 100
    ticker_df['%D'] = ticker_df['%K'].rolling(window=3).mean()

    return ticker_df

def algorithm(ticker_df):

    buy = ticker_df['Buy RSI'].unique()[0]
    sell = ticker_df['Sell RSI'].unique()[0]

    SMA_Small = int([x.split('_')[1] for x in ticker_df.columns if 'SMA' in x][0])
    SMA_Big = int([x.split('_')[1] for x in ticker_df.columns if 'SMA' in x][1])

    ticker_df['Signal'] = 0
    ticker_df['SMA_Signal'] = 0


    #buy condition 
    
    #these are how you check if you want the indicator to indicate a buy based on its variables
    ticker_df['RSI_Signal'] = np.where((ticker_df['RSI'] < buy), 1, 0) 
    ticker_df.loc[SMA_Small:, 'SMA_Signal'] = (ticker_df.loc[SMA_Small:, f'SMA_{SMA_Small}'] < ticker_df.loc[SMA_Small:, f'SMA_{SMA_Big}']).astype(int)
    ticker_df['BB_Signal'] = np.where(ticker_df['Lower Band'] > ticker_df['Close'], 1, 0)
    ticker_df['CMF_Signal'] = np.where(ticker_df['CMF'] < 0.1, 1, ticker_df['Signal'])
    ticker_df['STO_Signal'] = np.where(ticker_df['%D'] < ticker_df['%K'], 1, ticker_df['Signal'])

    #this is actually what the buy condition is
    ticker_df['Signal'] = np.where((ticker_df['RSI_Signal'] == 1) & (ticker_df['Volume Shift'] == 'increase'), 1, ticker_df['Signal'])

    #sell condition
    ticker_df['RSI_Signal'] = np.where(ticker_df['RSI'] > sell, -1, ticker_df['Signal'])
    ticker_df.loc[SMA_Small:, 'SMA_Signal'] = (ticker_df.loc[SMA_Small:, f'SMA_{SMA_Small}'] > ticker_df.loc[SMA_Small:, f'SMA_{SMA_Big}']).astype(int) - 2
    ticker_df['BB_Signal'] = np.where(ticker_df['Upper Band'] < ticker_df['Close'], -1, 0)
    ticker_df['CMF_Signal'] = np.where(ticker_df['CMF'] > 0.1, -1, ticker_df['CMF_Signal'])

    #this is actually where the sell condition is
    ticker_df['Signal'] = np.where((ticker_df['RSI_Signal'] == -1), -1, ticker_df['Signal'])

    return ticker_df

def backtest_data(ticker_df):

    #if you want to see the trades print buy_Sell and trades to get a better picture of whats being selected

    buy_sell = ticker_df[(ticker_df['Signal'] == 1) | (ticker_df['Signal'] == -1)]
    trades = []
    buy_flip = True
    last_signal = 1
    hold_last_rec = []
    records = []
    first_row = 1
    d = 'Date' if 'Date' in ticker_df.columns else 'Datetime'

    
    for x, y in buy_sell.iterrows():
        
        signal = y['Signal']
        #mark buy/sell on first time signal is indicated
        if buy_flip and signal == 1:
            trades.append(['Buy', y[d], y['Ticker'], y['Close']])
            buy_flip = False
        if not buy_flip and signal == -1:
            trades.append(['Sell', y[d], y['Ticker'], y['Close']])
            buy_flip = True 
    

    if len(trades) % 2 != 0: 
        trades.pop(-1)

    for i in range(0, len(trades), 2):
        range_start = ticker_df[(ticker_df[d] == trades[i][1]) & (ticker_df['Ticker'] == trades[i][2])].index[0]
        range_end = ticker_df[(ticker_df[d] == trades[i + 1][1]) & (ticker_df['Ticker'] == trades[i + 1][2])].index[0]

        cumulative_sum = ticker_df.loc[range_start:range_end, 'Percent Change'].sum()
        ticker_df.loc[range_start:range_end, 'Total Percent Change'] = cumulative_sum

        trades[i + 1].append(cumulative_sum)


    return trades

def display_plot(ticker_df, ind):

    colors = ['black', 'orange', 'green', 'red', 'blue', 'purple', 'brown', 'black', 'orange', 'green', 'red', 'blue', 'purple', 'brown']
    columns = [x for x in ticker_df.columns]
    d = 'Date' if 'Date' in ticker_df.columns else 'Datetime'
    RSI_bool = True
    BB_bool = True
    SMA_bool_1 = True
    SMA_bool_2 = True
    EMA_bool_1 = True
    EMA_bool_2 = True
    Signal_Bool = True
    STO_Bool = True

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_df[d], ticker_df['Close'], label='Close Price', color=colors.pop(), alpha=0.5) #default need to show stock

    for c in columns:

        if 'SMA' in c and 'Signal' not in c and 'SMA' in ind and (SMA_bool_1 or SMA_bool_2):
            SMA_Vals = c.split('_')[1]
            plt.plot(ticker_df[d], ticker_df[f'SMA_{SMA_Vals}'], label=f'SMA_{SMA_Vals}', color=colors.pop(), alpha=0.5)
            SMA_bool_2 = False if SMA_bool_1 == False else True 
            SMA_bool_1 = False

        if 'RSI' in c and RSI_bool and 'RSI' in ind:
            buy = ticker_df['Buy RSI'].unique()[0]
            sell = ticker_df['Sell RSI'].unique()[0]

            axis2 = plt.subplot(111)
            axis2.plot(ticker_df[d], ticker_df['RSI'], label='RSI', color=colors.pop(), alpha=0.5)
            axis2.axhline(sell, color='red', linestyle='--', label='Overbought')
            axis2.axhline(buy, color='green', linestyle='--', label='Oversold')
            axis2.grid()
            RSI_bool = False #one time run

        if 'Band' in c and BB_bool and 'BB' in ind:
            #plt.plot(ticker_df.index, ticker_df['Middle Band'], label='Middle Band', color=colors.pop(), alpha=0.5)
            plt.plot(ticker_df[d], ticker_df['Lower Band'], label='Lower Band', color=colors.pop(), alpha=0.5)
            plt.plot(ticker_df[d], ticker_df['Upper Band'], label='Upper Band', color=colors.pop(), alpha=0.5)
            BB_bool = False #one time run

        if 'EMA' in c and 'EMA' in ind and (EMA_bool_1 or EMA_bool_2):
            EMA_Vals = c.split('_')[1]
            plt.plot(ticker_df['Date'], ticker_df[f'EMA_{EMA_Vals}'], label=f'EMA{EMA_Vals}', color=colors.pop(), alpha=0.5)
            EMA_bool_2 = False if EMA_bool_1 == False else True 
            EMA_bool_1 = False
            
        if 'MACD' in c and 'MACD' in ind:
            plt.plot(ticker_df['Date'], ticker_df['MACD'], label='MACD', color=colors.pop(), alpha=0.5)
            plt.plot(ticker_df['Date'], ticker_df['MACD Line'], label='MACD Line', color=colors.pop(), alpha=0.5)

        if 'CMF' in c and 'CMF' in ind:
            axis2 = plt.subplot(111)
            axis2.plot(ticker_df['Date'], ticker_df['CMF'], label='CMF', color='green', alpha=0.7)
            axis2.axhline(0, color='black', linestyle='-', linewidth=1)  # Add a horizontal line at 0 for CMF
            axis2.set_ylabel('Chaikin Money Flow (CMF)', color='green')
            axis2.tick_params(axis='y', labelcolor='green')

            axis2.grid()
            
        if '%K' in c or '%D' in c and STO_Bool:
            if 'STO' in ind:
                plt.plot(ticker_df['Date'], ticker_df['%D'], label='D', color=colors.pop(), alpha=0.5)
                plt.plot(ticker_df['Date'], ticker_df['%K'], label='K', color=colors.pop(), alpha=0.5)
                STO_Bool = False

        if 'Signal' in c and Signal_Bool:
            buy_signals = ticker_df[ticker_df['Signal'] == 1]
            sell_signals = ticker_df[ticker_df['Signal'] == -1]

            plt.plot(buy_signals[d], buy_signals['Close'], '^', markersize=5, color='green', lw=0, label='Buy Signal')
            plt.plot(sell_signals[d], sell_signals['Close'], 'v', markersize=5, color='red', lw=0, label='Sell Signal')
            Signal_Bool = False
        
    plt.title('Buy/Sell Signals')
    plt.xlabel('Date')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.plot()
    plt.show()

    return 0

def run_indicators(ticker_df, iv):

    #ind_vars = [SMA_Roll_Win1, SMA_Roll_Win2, RSI_Period, RSI_Buy, RSI_Sell, BB_Period, BB_Multiplier, EMA_Roll_Win1, EMA_Roll_Win2, CMF_Win]

    SMA_results = SMA(ticker_df, iv[0], iv[1]) #df, rolling window 1 (days), rolling window 2 (days)
    RSI_results = RSI(ticker_df, iv[2], iv[3], iv[4]) #df, period (days), RSI BUY, RSI SELL
    BB_results = BB(ticker_df, iv[5], iv[6]) #df, period (days)
    EMA_Results = EMA(ticker_df, iv[7], iv[8]) #df, rolling window 1 (days), rolling window 2 (days)
    MACD_Results = MACD(ticker_df) #df
    CMF_Results = CMF(ticker_df, iv[9]) #df
    STO_Results = STO(ticker_df, 14)
    Signal_Results = algorithm(ticker_df)

    return ticker_df



#lmain function
def main():

    if len(sys.argv) == 1:
        print(intro_msg)
        ticker = input('Please enter a ticker: ')
        ticker = ticker.upper()
        year = input('Please enter a year: ')
        interval = input('Please enter an interval: ')

        file_path = f'historical_data/{ticker}/{year}/{interval}'
        if not os.path.isdir(file_path):
            split_month = input('Pulling data, would you like to split it by month? (y/n): ')
            subprocess.run(['python', f'historical.py', ticker, year, interval, split_month])

        by_month = input('Enter Month Number ## or \'y\' to use whole year: ')
        indicator = input('Please enter a indicator: ')

        match indicator:
            case 'RSI':
                ind_val1 = input('Please enter RSI_Period: ')
                ind_val2 = input('Please enter RSI Buy Value: ')
                ind_val3 = input('Please enter RSI Sell Value: ')

    else:
        ticker = sys.argv[1]
        year = sys.argv[2]
        interval = sys.argv[3]
        by_month = sys.argv[4]
        file_path = f'historical_data/{ticker}/{year}/{interval}'
        if not os.path.isdir(file_path):
            split_month = 'y' if by_month != 'y' else 'n'
            subprocess.run(['python', f'historical.py', ticker, year, interval, split_month])
        indicator = sys.argv[5] if len(sys.argv[5].split(',')) == 1 else sys.argv.split(',')
        ind_val1 = sys.argv[6] if len(sys.argv) > 6 else 0
        ind_val2 = sys.argv[7] if len(sys.argv) > 7 else 0
        ind_val3 = sys.argv[8] if len(sys.argv) > 8 else 0

    money = 1000
    algorithm_name = 'Default_Run'

    #indicator variables
    RSI_Period = ind_val1
    RSI_Buy = ind_val2
    RSI_Sell = ind_val3
    SMA_Roll_Win1 = 6
    SMA_Roll_Win2 = 12
    BB_Period = 20
    BB_Multiplier = 1.1
    EMA_Roll_Win1 = 20
    EMA_Roll_Win2 = 50
    CMF_Win = 3
    ind_vars = [SMA_Roll_Win1, SMA_Roll_Win2, int(RSI_Period), int(RSI_Buy), int(RSI_Sell), BB_Period, BB_Multiplier, EMA_Roll_Win1, EMA_Roll_Win2, CMF_Win]


    if by_month == 'y':
        file_path = f'historical_data/{ticker}/{year}/{interval}/{ticker}_{year}_{interval}.csv'
        year_data = read_historical_data(file_path) 
    else:
        file_path = f'historical_data/{ticker}/{year}/{interval}/split_month/{ticker}_{by_month}_{interval}.csv'
        year_data = read_historical_data(file_path) 

    d = 'Date' if 'Date' in year_data.columns else 'Datetime'

    tick = year_data
    tick['Percent Change'] = tick['Close'].pct_change() * 100
    tick['Volume Shift'] = tick['Volume'].diff().apply(lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no change'))
    tick['Volume Difference'] = tick['Volume'].diff()
    tick['Price Change'] = pd.to_numeric(tick['Close']).diff()
    tick['Time Diff'] = (pd.to_datetime(tick[d]).diff().dt.total_seconds() / 60) / 1000
    tick['Price Velocity'] = tick['Price Change'] / tick['Time Diff']

    tick = run_indicators(tick, ind_vars) #this generates finalized document to where you can calculate trades based off the numbers crunched


    #these are the results when to buy and sell stocks
    trades = backtest_data(tick)
    total_percent = sum([float(x[4]) for x in trades if x[0] == 'Sell'])
    total_trades = int(len(trades)) / 2
    final_money = money + (money * (total_percent / 100))
    tick.to_csv(f'analysis/{ticker}_{year}_{interval}.csv', index=False)
    
    display_plot(tick, ['SMA'])
    print(trades)

    all_trades = pd.DataFrame(columns=['Ticker', 'Close',  'Date', 'Buy/Sell'])
    for tr in trades:
        all_trades.loc[len(all_trades)] = [tr[2], tr[3], tr[1], tr[0]]

    all_trades.to_csv(f'run/{ticker}_{year}_{interval}_trades.csv', index=False)


    



main()