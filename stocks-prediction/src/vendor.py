import requests
import pandas as pd
from mlrun import get_or_create_ctx
import json

# Replace 'your_api_key_here' with your actual Twelve Data API key
api_key = '<api-key>'
symbol = 'AAPL'

#endpoint = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&apikey={api_key}'
#endpoint = f'https://api.twelvedata.com/time_series?symbol=AAPL,EUR/USD,ETH/BTC:Huobi,TRP:TSX&interval=1min&apikey={api_key}'
#endpoint = f'https://api.twelvedata.com/time_series?symbol=AAPL,GOOG,AMZN&interval=1min&apikey=<api-key>'
#endpoint = f'https://api.twelvedata.com/time_series?symbol=AAPL,GOOG,AMZN&interval=1min&apikey={api_key}'




def get_stocks(event):
    ctx = get_or_create_ctx(name="stocks-context")
    symbols = event['symbols']
    api_key = event['api_key']



    endpoint = f'https://api.twelvedata.com/time_series?symbol={symbols}&interval=1min&apikey={api_key}'
    json_response = requests.get(endpoint)

    json_string = json_response.text
    # Parse the JSON string into a Python dictionary
    data = json.loads(json_string)

    # Initialize a dictionary to hold the DataFrame for each stock
    stock_dataframes = {}

    for symbol, stock_info in data.items():
        # Convert the 'values' list (which contains the time series data for each stock) into a DataFrame
        df = pd.DataFrame(stock_info['values'])

        # Convert columns to appropriate data types, if necessary
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df['volume'] = df['volume'].astype(int)

        # Store the DataFrame in our dictionary
        stock_dataframes[symbol] = df


    ctx.logger.info("finished processing stocks")



    # return json.dumps(stock_dataframes)
    #
    # # Example usage: print the DataFrame for AAPL
    print(type(stock_dataframes))
    print(stock_dataframes)

get_stocks({'api_key' : '<api-key>','symbols' : 'AAPL,AMZN,GOOG'})

