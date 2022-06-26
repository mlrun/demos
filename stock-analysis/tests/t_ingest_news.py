import mlrun

project = mlrun.get_or_create_project(name='stocks', user_project=True, context="./")


STREAM_SHARDS = 1  # NO. of input stream shard
SERVING_FUNCTION_RESPONSE = 'SYNC'
MIN_REPLICAS = 1  # serving function minimum replicas
MAX_REPLICAS = 1  # serving function maximum replicas
SERVING_PROCESSING_TIME = 1  # time for the serving function to sleep
WORKER_AVAILIBALITY_TIMEOUT = 100
INGRESS_TIMEOUT = 150
USE_GPU = False  # whether the serving function will use GPU (if true - REMOTE_SCALE_RANGE will determine the consumption)



# RemoteStep configuration
REMOTE_STEP_WORKERS = 1
MAX_IN_FLIGHT = 1  # how many simulteneous events on each worker in the serving function
REMOTE_STEP_HTTP_TIMEOUT = 150
STOREY_QUEUE_SIZE = 8
WINDOW_ACK = (STOREY_QUEUE_SIZE+MAX_IN_FLIGHT) * STREAM_SHARDS


model_location = 'https://iguazio-sample-data.s3.amazonaws.com/models/model.pt'
scaled_function = mlrun.import_function('hub://sentiment_analysis_serving:development')
scaled_function.spec.min_replicas = MIN_REPLICAS
scaled_function.spec.max_replicas = MAX_REPLICAS
scaled_function.add_model('sentiment_analysis_model', model_path=model_location,
                          class_name='SentimentClassifierServing')

scaled_function.with_http(workers=REMOTE_STEP_WORKERS, gateway_timeout=INGRESS_TIMEOUT,
                          worker_timeout=WORKER_AVAILIBALITY_TIMEOUT)

env_vars = {"SERVING_FUNCTION_TIME_TO_SLEEP": SERVING_PROCESSING_TIME}

scaled_function.set_envs(env_vars)
scaled_function.spec.readiness_timeout = 300
#address = scaled_function.deploy()
address = "https://stocks-avia-sentiment-analysis-serving-stocks-avia.default-tenant.app.dev39.lab.iguazeng.com/"

# %%

# mlrun: start-code

import yahoo_fin.stock_info as si
import yahoo_fin.news as ynews
from dateutil import parser
import pandas as pd
import json
import requests
from storey import MapClass, Event
import string
import mlrun

def print_event(event):
    sentiment = event.body['outputs']['predictions'][0]/2
    return_event = Event(body={'sentiment': sentiment}, key=event.body['outputs']['meta_data']['ticker'], time=event.body['outputs']['meta_data']['Datetime'])
    print(return_event)
    return return_event


def wrap_event(event):
    print(event)
    wrapped_event = {'meta_data': {'ticker': event['ticker'], 'Datetime': event['Datetime']}, 'inputs': [event['summary']]}
    print(wrapped_event)
    return wrapped_event

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
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
        df_copy = news_df[['title', 'summary', 'link', 'published']].copy()
        df_copy['ticker'] = ticker
        df_copy['Datetime'] = df_copy['published'].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d %H:%M:%S'))
        df_copy['summary'] = df_copy['summary'].apply(lambda x: remove_punctuation(x))
        df_copy['title'] = df_copy['title'].apply(lambda x: remove_punctuation(x))
        tickers_news.append(df_copy)
    df = pd.concat(tickers_news).reset_index(drop=True)
    return json.loads(df.to_json(orient='records'))


# fn = mlrun.import_function('hub://sentiment_analysis_serving:development')
# mod = mlrun.function_to_module(fn, workdir='./')
#
#
# mod.SentimentClassifierServing

import torch
from mlrun.serving.remote import RemoteStep

import mlrun.feature_store as fstore
from mlrun.feature_store.steps import DateExtractor, MapValues
import yahoo_fin.stock_info as si

# creating feature set
news_set = fstore.FeatureSet("stocks_news",
                             entities=[fstore.Entity("ticker")],
                             timestamp_key='Datetime',
                             description="stocks news feature set")

# how many tickers data we ingest (make sure same number used for ingesting news)
# n_tickers = 4

news_set.graph \
    .to(name='get_news', handler='get_news') \
    .to("storey.steps.Flatten", name="flatten_news") \
    .to(name='wrap_event', handler='wrap_event') \
    .to(RemoteStep(name="remote_scale", url=address, method="POST", max_in_flight=MAX_IN_FLIGHT,
                   timeout=REMOTE_STEP_HTTP_TIMEOUT)) \
    .to(name='print_event', handler='print_event',full_event=True) \
    #     .to(class_name = mod.SentimentClassifierServing() ,name= "sentiment_analysis_model", model_path='https://iguazio-sample-data.s3.amazonaws.com/models/model.pt',
#         function='')\


#     .to("sentiment_analysis", "sentiment_analysis_model",full_event=True)\

news_set.set_targets(with_defaults=True)
#news_set.plot(rankdir="LR", with_targets=True)

# %% md

## Dummy ingestion, Deploying ingestion service and getting ingestion endpoint

# %%

# ingesting dummy (A MUST)
import os
import datetime

name = os.environ['V3IO_USERNAME']
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

fstore.ingest(news_set,
              pd.DataFrame.from_dict({'ticker': [name],
                                      'Datetime': now,
                                      'n_stocks': 1}))

