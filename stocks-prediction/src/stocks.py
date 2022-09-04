#mlrun: start-code

import yahoo_fin.stock_info as si
import datetime
import yfinance as yf
import pandas as pd
import json
from storey import Event

def get_stocks(event):
    '''
    event: dict with the following keys:
    start_delta - start collecting the data days back
    end_delta - collect data untill days back
    interval - interval of collected stocks data
    n_stocks - how many stocks to collect
    '''
    # getting stock names
    tickers = si.tickers_sp500()[:event['n_stocks']]
    # time deltas to scrape data
    start = datetime.datetime.now()-datetime.timedelta(event['start_delta'])
    end = datetime.datetime.now()-datetime.timedelta(event['end_delta'])
    interval = event['interval']
    
    # collecting data from yfinance
    return_list = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
        hist['ticker'] = ticker
        hist['ticker2onehot'] = ticker
        return_list.append(hist)
        
    # some data manipulations
    df = pd.concat(return_list).reset_index().drop(axis=1,columns=['Dividends','Stock Splits'])
    df['Datetime']= df['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    
    return json.loads(df.to_json(orient='records'))

def gen_event_key(event): # since using nosql as target, each event must have its key - therefore this step is needed !
    return Event(event.body,key=event.body['ticker'])

#mlrun: end-code