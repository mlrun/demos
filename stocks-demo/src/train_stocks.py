
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mlrun.feature_store as fstore
import datetime
import mlrun.frameworks.pytorch as mlrun_pytorch
import pickle
from mlrun.frameworks.pytorch import PyTorchMLRunInterface, PyTorchModelHandler, PyTorchModelServer
import os

def accuracy(y_pred, y_true) -> float:
    """
    Accuracy metric.

    :param y_pred: The model's prediction.
    :param y_true: The ground truth.

    :returns: The accuracy metric value.
    """
    return 1 - (torch.norm(y_true - y_pred) / y_true.size()[0]).item()

# creating dataset 
class stocks_dataset(Dataset):
    def __init__(self, vector_name = 'stocks', seq_size = 5, start_time=None, end_time=None):
        start_time = datetime.datetime.now()-datetime.timedelta(start_time)
        end_time = datetime.datetime.now()-datetime.timedelta(end_time)
        train_dataset = fstore.get_offline_features(vector_name, entity_timestamp_column='Datetime',  with_indexes=True, start_time=start_time, end_time=end_time)
        price_cols = ['Open','High','Low','Close']
        self.df = train_dataset.to_dataframe().reset_index(drop=False)
        self.df.fillna(value=1,inplace=True)
        self.normalized_df = self.df.copy()
        self.tickers = self.df['ticker'].unique()
        self.data = []
        self.labels = []
        self.normalized_df[price_cols] = (self.normalized_df[price_cols] - self.normalized_df[price_cols].mean()) / self.normalized_df[price_cols].std()
        self.normalized_df['Volume'] = (self.normalized_df['Volume'] - self.normalized_df['Volume'].mean()) / self.normalized_df['Volume'].std()
        for ticker in self.tickers:
            ticker_df = self.normalized_df[self.normalized_df['ticker'] == ticker].sort_values(by='Datetime').drop(['ticker','Datetime'],axis=1)
            for i in range(0,ticker_df.shape[0]-seq_size-1):
                self.data.append(ticker_df[i:i+seq_size].values.tolist())
                self.labels.append(ticker_df.iloc[i+seq_size].values.tolist())
        
        self.data = torch.tensor(self.data).detach()
        self.labels = torch.tensor(self.labels, dtype=torch.float).detach()
        
    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
        
        
    def __len__(self):
        return len(self.data)
    
    
class Model(nn.Module):
    def __init__(self, input_size=10, output_size=10, hidden_dim=10, n_layers=4, batch_size=32, seq_size=5):
        super(Model, self).__init__()
        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_size=seq_size
        
        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        
        
        self.fc = nn.Linear(hidden_dim*seq_size*batch_size, input_size*batch_size)
        # Initializing hidden state for first input using method defined below
        self.hidden = self.init_hidden()
    
    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, self.hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1)
        out = self.fc(out)
        out = out.view(self.batch_size,-1)
        return out
    
    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        return hidden
    
    
    
def train_stocks(context,
          vector_name = 'stocks',
          start_time=7,
          end_time=0,
          batch_size = 32,
          hidden_dim=10,
          n_layers=4,
          seq_size=5,
          epochs=3, 
         ):
    
    dataset = stocks_dataset(vector_name, seq_size, start_time, end_time) 
    training_set = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    input_size = output_size = dataset.data[0][0].shape[0]
    # creating the model
    model = Model(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers,batch_size=batch_size,seq_size=seq_size)
    
#     mlrun_torch = PyTorchMLRunInterface(model=model, context=context).add_auto_logging_callbacks()
    # Initialize the optimizer:
    optimizer = torch.optim.Adam(lr=0.0001, params=model.parameters())
    criterion = nn.MSELoss()
    
    # training with mlrun's torch interface
    mlrun_pytorch.train(model=model,
                        training_set=training_set,
                        validation_set=training_set,
                        training_iterations=35,
                        loss_function=criterion,
                        optimizer=optimizer,
                        epochs=epochs,
                        use_cuda = False,
                        use_horovod = False,
                        model_name = 'pytorch_stocks_model',
                        custom_objects_map={"train_stocks.py": "Model"},
                        custom_objects_directory='/User/junkyard/stocks_recreate/src',
                        metric_functions=[accuracy],
                        auto_log=True,
                        )
