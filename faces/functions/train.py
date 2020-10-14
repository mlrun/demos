from mlrun import get_or_create_ctx
import torch.nn as nn
import torch
import os
import random
import v3io_frames as v3f
import importlib.util
from pickle import dump
from functions.params import Params


def read_encodings_table(frames_url, token, container='faces', table='encodings'):
    client = v3f.Client(address=frames_url, token=token, container=container)
    encoding_df = client.read(backend="kv", table=table, reset_index=False, filter='label != -1')
    return encoding_df


def train(context, model_name='model.bst', cuda=True):
    params = Params()
    params.set_params_from_context(context)
    if cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            context.logger.info(f"Running on cuda device: {device}")
        else:
            device = torch.device("cpu")
            context.logger.info("Requested running on cuda but no cuda device available.\nRunning on cpu")
    else:
        device = torch.device("cpu")

    # prepare data from training
    context.logger.info('Client')

    data_df = read_encodings_table(params.frames_url, params.token)

    X = data_df[['c' + str(i).zfill(3) for i in range(128)]].values
    y = data_df['label'].values

    n_classes = len(set(y))

    X = torch.as_tensor(X, device=device)
    y = torch.tensor(y, device=device).reshape(-1, 1)

    input_dim = 128
    hidden_dim = 64
    output_dim = n_classes

    spec = importlib.util.spec_from_file_location('models', params.models_path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)

    model = models.FeedForwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    model = model.double()

    # define loss and optimizer for the task
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the network
    n_iters = X.size(0) * 5
    for i in range(n_iters):
        r = random.randint(0, X.size(0) - 1)
        optimizer.zero_grad()
        out = model(X[r]).reshape(1, -1)
        loss = criterion(out, y[r])
        loss.backward()
        optimizer.step()

    context.logger.info('Save model')
    # saves and logs model into mlrun context
    dump(model._modules, open(model_name, 'wb'))
    context.log_artifact('model', src_path=model_name, target_path=params.artifacts_path+model_name,
                         labels={'framework': 'Pytorch-FeedForwardNN'})
    os.remove(model_name)


if __name__ == '__main__':
    ctx = get_or_create_ctx('train model')
    train(ctx, "data")
