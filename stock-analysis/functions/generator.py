from abc import ABC, abstractmethod

import pandas
import yahoo_fin.stock_info as si
import datetime
import yfinance as yf
import pandas as pd
import time
import yahoo_fin.news as ynews
from mlrun import get_or_create_ctx
from dateutil import parser
import string
import pathlib


class IGenerator(ABC):

    @abstractmethod
    def generate(self):
        pass

    def create_file_name(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        file_name = path + time.strftime("%Y%m%d-%H%M%S") + '-' + self.__class__.__name__ + '.csv'
        return file_name


class StocksGenerator(IGenerator, ABC):

    def __init__(self):
        self.ctx = get_or_create_ctx(name='stocks-generator')

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
            return_list.append(hist)

        # some data manipulations
        df: pandas.DataFrame = pd.concat(return_list).reset_index().drop(axis=1, columns=['Dividends', 'Stock Splits'])
        print(f"downloaded {len(tickers)} stocks data with size of {df.shape}")
        df['Datetime'] = df['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        file_name = self.create_file_name(path)
        self.ctx.logger.info("writing file to path {}".format(file_name))
        df.to_csv(file_name, index=False)


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


class NewsGenerator(IGenerator):

    def __init__(self):
        self.ctx = get_or_create_ctx(name='news-generator')

    def generate(self, number_of_stocks, path):
        """
        event: dict with the following keys:
        n_stocks - how many stocks to collect
        """
        tickers = si.tickers_sp500()[:number_of_stocks]
        tickers_news = []
        for ticker in tickers:
            news = ynews.get_yf_rss(ticker=ticker)
            news_df = pd.DataFrame(news)
            df_copy = news_df[['title', 'summary', 'link', 'published']].copy()
            df_copy['ticker'] = ticker
            df_copy['Datetime'] = df_copy['published'].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d %H:%M:%S'))
            df_copy['summary'] = df_copy['summary'].apply(lambda x: remove_punctuation(x))
            df_copy['title'] = df_copy['title'].apply(lambda x: remove_punctuation(x))
            tickers_news.append(df_copy)
        df = pd.concat(tickers_news).reset_index(drop=True)
        file_name = self.create_file_name(path)
        self.ctx.logger.info("writing file to path {}".format(file_name))
        df.to_csv(file_name, index=False)
