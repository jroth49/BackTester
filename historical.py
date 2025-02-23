import pandas as pd
import yfinance as yf
import logging
import os
import sys

logging.basicConfig(level=logging.CRITICAL)
pd.options.mode.chained_assignment = None
intro_msg = r'''
⠀⠀⣀⠠⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠉⠙⠛⠿⠶⠶⣖⣢⣔⣶⡄⠀⠀⠀⠀⠀⢤⣀⣀⣀⣀⣀⣠⣤⣴⣶⡲⠆⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀
⢀⣀⣤⣤⡖⠛⣭⡉⠓⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⢴⠒⠒⠒⠦⢄⡀⠀⠀
⠙⣇⠀⠀⣷⡀⠛⠃⢀⡇⠙⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡿⠟⠉⢿⠀⠿⠃⢀⡇⠈⣱⣦
⠀⠈⠳⣄⠀⠙⠓⠋⠉⣀⡔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣦⡀⠘⠷⠦⠶⠋⢀⡴⠋⠀
⠀⠀⠀⠈⠓⠦⠤⠖⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠷⣦⣤⣶⠞⠉⠀⠀⠀

Welcome to Stock Market Data Analysis
'''

def convert_month(number):
    month = 'ERR'
    match number:
        case '01':
            month = 'jan'
        case '02':
            month = 'feb'
        case '03':
            month = 'mar'
        case '04':
            month = 'apr'
        case '05':
            month = 'may'
        case '06':
            month = 'jun'
        case '07':
            month = 'jul'
        case '08':
            month = 'aug'
        case '09':
            month = 'sep'
        case '10':
            month = 'oct'
        case '11':
            month = 'nov'
        case '12':
            month = 'dec'
    return month

def get_historical_data(tickers, int, st, nd, split_month):

    output_df = pd.DataFrame()
    year = st.split('-')[0] #st = '2000-01-01' => ['2000', '01', '01'] => '2000'

    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            ticker_history = ticker.history(interval = int, start = st, end = nd)
            ticker_history.insert(0, 'Ticker', t)
            output_df = pd.concat([output_df, ticker_history])
            output_df.loc[output_df['Ticker'] == t, 'Sector'] = ticker.info.get('sectorKey')
        except Exception as e:
            print('Error ' + t + f' {e}...')
            print(st)
            print(nd)

        if split_month:
            month_num = st.split('_')[0].split('-')[1]
            output_df.to_csv(f'historical_data/{t}/{year}/{int}/split_month/{t}_{month_num}_{int}.csv')
        else:
            output_df.to_csv(f'historical_data/{t}/{year}/{int}/{t}_{year}_{int}.csv')

def main():

    if len(sys.argv) == 1:
        print(intro_msg)
    
        #Get ticker
        tickers = [input('Please enter the Ticker: ')]

        #Get Year Youd Like TO Pull
        year = input('Please enter the Year: ')
        
        #Get Interval
        interval = input('Please enter the Interval (“1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”): ')

        #Would you like it split by months
        split_month = input('Would you like the interval split by months? (y/n): ')
        split_month = True if split_month == 'y' else False

    else:
        tickers = [sys.argv[1]] if not len(sys.argv[1].split(',')) > 1 else sys.argv[1].split(',')
        year = sys.argv[2]
        interval = sys.argv[3]
        split_month = True if sys.argv[4] == 'y' else False
    
    #leap year
    leap_year = f'{year}-02-28' if int(year) % 4 != 0 else f'{year}-02-29'
    month_split = [
        [f'{year}-01-01', f'{year}-01-31'],
        [f'{year}-02-01', leap_year],
        [f'{year}-03-01', f'{year}-03-31'],
        [f'{year}-04-01', f'{year}-04-30'],
        [f'{year}-05-01', f'{year}-05-31'],
        [f'{year}-06-01', f'{year}-06-30'],
        [f'{year}-07-01', f'{year}-07-31'],
        [f'{year}-08-01', f'{year}-08-31'],
        [f'{year}-09-01', f'{year}-09-30'],
        [f'{year}-10-01', f'{year}-10-31'],
        [f'{year}-11-01', f'{year}-11-30'],
        [f'{year}-12-01', f'{year}-12-31'],
    ]

    #directory prep
    for t in tickers:
        if not os.path.isdir(f'historical_data/{t}'):
            os.mkdir(f'historical_data/{t}')

        if not os.path.isdir(f'historical_data/{t}/{year}'):
            os.mkdir(f'historical_data/{t}/{year}')
        
        if not os.path.isdir(f'historical_data/{t}/{year}/{interval}'):
            os.mkdir(f'historical_data/{t}/{year}/{interval}')

        if split_month:
            if not os.path.isdir(f'historical_data/{t}/{year}/{interval}/split_month'):
                os.mkdir(f'historical_data/{t}/{year}/{interval}/split_month')

    if split_month:
        for m in month_split:
            get_historical_data(tickers, interval, m[0], m[1], True)
    else:
        get_historical_data(tickers, interval, f'{year}-01-31', f'{year}-12-31', False)


main()
