from functions.read_news import handler, init_context
from nuclio_sdk import Context, Logger
import os
import json

# create nuclio empty context for testing
def create_context():
    logger = Logger(level=20)
    ctx = Context(logger=logger)
    return ctx


def test_read_news():
    os.environ['V3IO_FRAMESD'] = 'https://framesd.default-tenant.app.dev8.lab.iguazeng.com'
    os.environ['TOKEN'] = 'bc93c3d4-94a9-4650-a79a-e162b351a42a'
    os.environ['SENTIMENT_MODEL_ENDPOINT'] = 'http://stocks-avia-sentiment-analysis-serving-stocks-avia.default-tenant.app.dev8.lab.iguazeng.com'
    # create a test event and invoke the function locally
    ctx = create_context()
    init_context(ctx)
    handler(ctx, None)


def test_load_resp():
    resp_text = '{"id": "476d7cf8-d3d1-4c06-a372-17baae42596e", "model_name": "bert_classifier_v1", "outputs": [2, 0, 0, 2, 0, 1, 0, 1, 0, 1, 1, 1, 0]}'
    js = json.loads(resp_text)
    print(js)

