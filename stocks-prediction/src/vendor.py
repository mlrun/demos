import requests
import pandas as pd
from mlrun import get_or_create_ctx
import json
from storey import Event


def get_stocks(event):

    ctx = get_or_create_ctx(name="stocks-context")
    stocks_list = event['stocks_list']
    api_key = event['api_key']

    endpoint = f'https://api.twelvedata.com/time_series?symbol={stocks_list}&interval=1min&apikey={api_key}'
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

        # Add the dictionary key as a column in each DataFrame
        for key, df in stock_dataframes.items():
            df['ticker'] = key  # 'Source' is the new column for the dictionary key

        ctx.logger.info("finished processing stocks")

        combined_df = pd.concat(stock_dataframes.values(), ignore_index=True)

        # # Example usage: print the DataFrame for AAPL
        return_df = pd.DataFrame.from_dict(data=combined_df)

        return json.loads(return_df.to_json(orient='records'))


def gen_event_key(event):  # since using nosql as target, each event must have its key - therefore this step is needed !
    return Event(event.body, key=event.body['ticker'])