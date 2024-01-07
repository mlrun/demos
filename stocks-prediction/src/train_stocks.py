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
import torch
from torch.utils.data import DataLoader, Dataset
import mlrun.feature_store as fstore
import datetime
import mlrun.frameworks.pytorch as mlrun_pytorch
import pandas as pd
from mlrun import get_or_create_ctx


def accuracy(y_pred, y_true) -> float:
    """
    Accuracy metric.

    :param y_pred: The model's prediction.
    :param y_true: The ground truth.

    :returns: The accuracy metric value.
    """
    return 1 - (torch.norm(y_true - y_pred) / y_true.size()[0]).item()


# creating dataset
class StocksDataset(Dataset):
    def __init__(self, vector_name='stocks', seq_size=5, start_time=None, end_time=None):
        start_time = datetime.datetime.now() - datetime.timedelta(start_time)
        end_time = datetime.datetime.now() - datetime.timedelta(end_time)
        train_dataset = fstore.FeatureVector.get_offline_features(vector_name, timestamp_for_filtering='Datetime', with_indexes=True,
                                                    start_time=start_time, end_time=end_time)
        price_cols = ['Open', 'High', 'Low', 'Close']
        self.df = train_dataset.to_dataframe().reset_index(drop=False)
        self.df.fillna(value=1, inplace=True)
        self.normalized_df = self.df.copy()
        self.tickers = self.df['ticker'].unique()
        self.data = []
        self.labels = []
        price_series = pd.concat([self.normalized_df[col] for col in price_cols])
        price_std = price_series.std()
        price_mean = price_series.mean()

        self.normalized_df[price_cols] = (self.normalized_df[price_cols] - price_mean) / price_std
        self.normalized_df['Volume'] = (self.normalized_df['Volume'] - self.normalized_df['Volume'].mean()) / \
                                       self.normalized_df['Volume'].std()
        for ticker in self.tickers:
            ticker_df = self.normalized_df[self.normalized_df['ticker'] == ticker].sort_values(by='Datetime').drop(
                ['ticker', 'Datetime'], axis=1)
            for i in range(0, ticker_df.shape[0] - seq_size - 1):
                self.data.append(ticker_df[i:i + seq_size].values.tolist())
                self.labels.append(ticker_df.iloc[i + seq_size]['Close'])

        self.data = torch.tensor(self.data).detach()
        self.labels = torch.tensor(self.labels, dtype=torch.float).detach()

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class Model(torch.nn.Module):
    def __init__(self, input_size=16, output_size=1, hidden_dim=2, n_layers=1, batch_size=1, seq_size=5):
        super(Model, self).__init__()
        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_size = seq_size

        # Defining the layers
        # RNN Layer
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim * seq_size * batch_size, output_size * batch_size)

        # Initializing hidden state for first input using method defined below
        self.hidden = self.init_hidden()

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, self.hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1)
        out = self.fc(out)
        return out

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        return hidden


def handler(vector_name='stocks',
            start_time=59,
            end_time=0,
            batch_size=1,
            hidden_dim=2,
            n_layers=1,
            seq_size=5,
            epochs=3,
            model_filepath=''
            ):
    context = get_or_create_ctx(name='train-context')
    dataset = StocksDataset(vector_name, seq_size, start_time, end_time)
    training_set = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    input_size = dataset.data[0][0].shape[0]
    output_size = 1
    # creating the model
    model = Model(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers,
                  batch_size=batch_size, seq_size=seq_size)

    # Initialize the optimizer:
    optimizer = torch.optim.Adam(lr=0.0001, params=model.parameters())
    criterion = torch.nn.MSELoss()

    # attaching run_id to model tag
    model_tag = context.uid

    # training with mlrun's torch interface
    mlrun_pytorch.train(model=model,
                        training_set=training_set,
                        validation_set=training_set,
                        loss_function=criterion,
                        optimizer=optimizer,
                        epochs=epochs,
                        use_cuda=False,
                        use_horovod=False,
                        model_name='stocks_model',
                        custom_objects_map={"train_stocks.py": "Model"},
                        custom_objects_directory=model_filepath,
                        metric_functions=[accuracy],
                        mlrun_callback_kwargs={"log_model_tag": model_tag},
                        auto_log=True,
                        context=context
                        )
