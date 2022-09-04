import yahoo_fin.stock_info as si
import yahoo_fin.news as ynews
from dateutil import parser
import pandas as pd 
import json
import requests
from storey import MapClass, Event
import string
import mlrun


def wrap_event(event):
    wrapped_event = {'meta_data': {'ticker': event['ticker'],
                                   'Datetime': event['Datetime'],
                                   'published': event['published'],
                                   'summary':event['summary'],
                                   'title': event['title']},
                     'inputs': [event['summary']]}
    
    return wrapped_event

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def get_news(event):
    '''
    event: dict with the following keys:
    n_stocks - how many stocks to collect
    '''
    tickers = si.tickers_sp500()[:event['n_stocks']]
    tickers_news = []
    for ticker in tickers:
        news = ynews.get_yf_rss(ticker=ticker)
        news_df = pd.DataFrame(news)
        df_copy = news_df[['title','summary','link','published']].copy()
        df_copy['ticker'] = ticker
        df_copy['published'] = df_copy['published'].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d %H:%M:%S'))
        df_copy['Datetime'] = df_copy['published'].copy()
        df_copy['summary'] = df_copy['summary'].apply(lambda x:remove_punctuation(x))
        df_copy['title'] = df_copy['title'].apply(lambda x:remove_punctuation(x))
        tickers_news.append(df_copy)
    df = pd.concat(tickers_news).reset_index(drop=True)
    return json.loads(df.to_json(orient='records'))

class sentiment_analysis():
    def __init__(self,address):
        self.address= address    
        
    def do(self,event):
        response = json.loads(requests.put(self.address + "v2/models/sentiment_analysis_model/predict",
                                                     json=json.dumps(event.body)).text)
        
        sentiment_event = event.body['meta_data']
        sentiment_event['sentiment'] = response['outputs']['predictions'][0]/2 # so it'll be 0 for neg, 0.5 for neutral and 1 for pos
        return Event(sentiment_event,key=sentiment_event['ticker'], time=sentiment_event['Datetime'])