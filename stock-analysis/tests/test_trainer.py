import mlrun
from mlrun import get_or_create_ctx
from functions.generator import *
from mlrun import code_to_function
from src.train_stocks import train_stocks


def test_train():
    ctx = get_or_create_ctx(name='train-context')
    train_stocks(context=ctx,model_filepath='.')

def test_train_stocks():

