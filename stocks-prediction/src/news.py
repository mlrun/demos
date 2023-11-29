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
import yahoo_fin.news as ynews
from dateutil import parser
import pandas as pd
import json
import requests
from storey import Event
import string
from transformers import pipeline
from mlrun import get_or_create_ctx


def wrap_event(event):
    wrapped_event = {'meta_data': {'ticker': event['ticker'],
                                   'Datetime': event['Datetime'],
                                   'published': event['published'],
                                   'summary': event['summary'],
                                   'title': event['title']},
                     'inputs': [event['summary']]}

    return wrapped_event


def remove_punctuation(text):
    punctuations = "".join([i for i in text if i not in string.punctuation])
    return punctuations


def convert_sentiment_to_int(sentiment):
    if sentiment[0]['label'] == 'NEGATIVE':
        return 0
    if sentiment[0]['label'] == 'POSITIVE':
        return 1
    else:
        return 2


def get_news(event):
    """
    event: dict with the following keys:
    n_stocks - how many stocks to collect
    """
    tickers = si.tickers_sp500()[:event['n_stocks']]
    tickers_news = []
    for ticker in tickers:
        news = ynews.get_yf_rss(ticker=ticker)
        news_df = pd.DataFrame(news)
        df_copy = news_df[['title', 'summary', 'link', 'published']].copy()
        df_copy['ticker'] = ticker
        df_copy['published'] = df_copy['published'].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d %H:%M:%S'))
        df_copy['Datetime'] = df_copy['published'].copy()
        df_copy['summary'] = df_copy['summary'].apply(lambda x: remove_punctuation(x))
        df_copy['title'] = df_copy['title'].apply(lambda x: remove_punctuation(x))
        tickers_news.append(df_copy)
    df = pd.concat(tickers_news).reset_index(drop=True)
    return json.loads(df.to_json(orient='records'))


class HuggingSentimentAnalysis:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.ctx = get_or_create_ctx("news-context")

    def get_sentiment(self, event):
        prediction = self.sentiment_pipeline(event.body['inputs'])
        event.body = event.body['meta_data']
        int_prediction = convert_sentiment_to_int(prediction)
        event.body['sentiment'] = int_prediction
        event.key = event.body['ticker']
        self.ctx.logger.info(event.body)
        return event
