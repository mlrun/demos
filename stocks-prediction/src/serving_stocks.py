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
import mlrun
import torch
from cloudpickle import load
from typing import List
import numpy as np
import warnings
import cloudpickle
import mlrun.feature_store as fstore
import pandas as pd
from mlrun.frameworks.pytorch import PyTorchModelServer
import datetime
import v3io_frames as v3f
import os
import time

warnings.filterwarnings("ignore")

def preprocess(event):
    vector_name = event['vector_name']
    start_time = datetime.datetime.now()-datetime.timedelta(event['start_time'])
    end_time = datetime.datetime.now()-datetime.timedelta(event['end_time'])
    seq_size = event['seq_size']
    
    train_dataset = fstore.get_offline_features(vector_name, timestamp_for_filtering='Datetime',  with_indexes=True, start_time=start_time, end_time=end_time)
    price_cols = ['Open','High','Low','Close']
    df = train_dataset.to_dataframe().reset_index(drop=False)
    df.fillna(value=1,inplace=True)
    normalized_df = df.copy()

    tickers = df['ticker'].unique()
    data = []
    labels = []
    tickers_list = []
    datetimes = []

    price_series = pd.concat([normalized_df[col] for col in price_cols])
    price_std = price_series.std()
    price_mean = price_series.mean()
        
    normalized_df[price_cols] = (normalized_df[price_cols] - price_mean) / price_std
    normalized_df['Volume'] = (normalized_df['Volume'] - normalized_df['Volume'].mean()) / normalized_df['Volume'].std()
    
    for ticker in tickers:
        ticker_df = normalized_df[normalized_df['ticker'] == ticker].sort_values(by='Datetime',ascending=False)
        timestamp = ticker_df.iloc[0]['Datetime']
        ticker_df = ticker_df.drop(['ticker','Datetime'],axis=1)
        data.append(ticker_df[1:seq_size+1].values.tolist())
        labels.append(ticker_df.iloc[0]['Close'])
        tickers_list.append(ticker)
        datetimes.append(timestamp)

    data = torch.tensor(data).detach()
    labels = torch.tensor(labels, dtype=torch.float).detach()
    price_series = pd.concat([normalized_df[col] for col in price_cols])
    
    event['columns'] = list(normalized_df.drop(['ticker','Datetime'],axis=1,inplace=False).columns)
    event['price_mean'] = price_mean
    event['price_std'] = price_std
    event['volume_mean'] = normalized_df['Volume'].mean()
    event['volume_std'] = normalized_df['Volume'].std()
    event['tickers'] = tickers_list
    event['datetimes'] = datetimes
    event['inputs'] = data.tolist()
    event['labels'] = labels.tolist()
    return event

def postprocess(event):
    df = pd.DataFrame(data=event['outputs']['results'],columns=['prediction'])
    df['datetime'] = event['outputs']['datetimes']
    df['tickers'] = event['outputs']['tickers']
    df['true'] = event['outputs']['labels']
    df['prediction'] = (df['prediction']*event['outputs']['price_std']) + event['outputs']['price_mean']
    df['true'] = (df['true']*event['outputs']['price_std']) + event['outputs']['price_mean']
    df2 = df.copy()
    df['datetime'] = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # writing to tsdb
    framesd = os.getenv("V3IO_FRAMESD",'framesd:8081')
    client = v3f.Client(framesd, container=os.getenv('V3IO_CONTAINER', 'projects'))
    kv_table_path = '/stocks-'+ os.environ['V3IO_USERNAME'] + '/artifacts/stocks_prediction'
    client.write('kv', kv_table_path, dfs=df, index_cols=['datetime','tickers'])
    print(f'writing prediction to kv at projects{kv_table_path} ...')
    return [df.values.tolist(),list(df.columns)]

class StocksModel(PyTorchModelServer):
    
    def predict(self, body: dict) -> List:
        all_results = []
        """Generate model predictions from sample."""
        for feats in body['inputs']:
            feats = torch.tensor(feats).reshape(1,5,-1)
            result: np.ndarray = self.model(feats)
            all_results.append(result.tolist()[0])
        body['results'] = all_results
        return body
from mlrun.runtimes import nuclio_init_hook
def init_context(context):
    nuclio_init_hook(context, globals(), 'serving_v2')

def handler(context, event):
    return context.mlrun_handler(context, event)
