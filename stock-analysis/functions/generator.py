from abc import ABC, abstractmethod

import pandas
import yahoo_fin.stock_info as si
import datetime
import yfinance as yf
import pandas as pd
import json
import time
from mlrun import get_or_create_ctx


class IGenerator(ABC):

    @abstractmethod
    def generate(self):
        pass

    def create_file_name(self):
        file_name = time.strftime("%Y%m%d-%H%M%S") + '-' + self.__class__.__name__ + '.csv'
        return file_name


class StocksGenerator(IGenerator, ABC):

    def __init__(self):
        self.ctx = get_or_create_ctx()

    def generate(self, number_of_stocks, start_delta, end_delta, interval, path):
        """
        event: dict with the following keys:
        start_delta - start collecting the data days back
        end_delta - collect data untill days back
        interval - interval of collected stocks data
        n_stocks - how many stocks to collect
        """
        # getting stock names
        tickers = si.tickers_sp500()[:number_of_stocks]
        # time deltas to scrape data
        start = datetime.datetime.now() - datetime.timedelta(start_delta)
        end = datetime.datetime.now() - datetime.timedelta(end_delta)
        interval = interval

        # collecting data from yfinance
        return_list = []
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
            hist['ticker'] = ticker
            hist['ticker2onehot'] = ticker
            return_list.append(hist)

        # some data manipulations
        df: pandas.DataFrame = pd.concat(return_list).reset_index().drop(axis=1, columns=['Dividends', 'Stock Splits'])
        print(f"downloaded {len(tickers)} stocks data with size of {df.shape}")
        df['Datetime'] = df['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        file_name = self.create_file_name()
        self.ctx.logger.info("writing file to path {}".format(file_name))
        df.to_csv(file_name, index=False)
        return json.loads(df.to_json(orient='records'))


class NewsGenerator(IGenerator):

    def generate(self):
        return
