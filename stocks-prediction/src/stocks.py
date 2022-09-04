# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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