import nuclio_sdk
from functions.read_news import handler, init_context
from mlrun import get_or_create_ctx , run_local , new_task, new_function, code_to_function
import os
import v3io_frames as v3f


base_task = new_task(params={})


def test_news_reader():
    ctx = nuclio_sdk.Context()
    init_context(ctx)
    handler(ctx, None)


def test_create_tsdb():
    frames_uri = os.getenv('V3IO_FRAMESD', 'framesd:8081')
    client = v3f.Client(frames_uri, container=os.getenv('V3IO_CONTAINER', 'bigdata'))
    client.create(backend='tsdb', table='stocks/stocks_tsdb', rate='1/s', if_exists=1)

