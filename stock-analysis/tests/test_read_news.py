from functions.read_news import handler, init_context
from nuclio_sdk import Context, Logger
import os


# create nuclio empty context for testing
def create_context():
    logger = Logger(level=20)
    ctx = Context(logger=logger)
    return ctx


def test_read_news():
    os.environ['V3IO_FRAMESD'] = 'https://framesd.default-tenant.app.dev8.lab.iguazeng.com'
    os.environ['TOKEN'] = 'bc93c3d4-94a9-4650-a79a-e162b351a42a'
    os.environ['SENTIMENT_MODEL_ENDPOINT'] = 'http://stocks-sentiment-analysis-serving-stocks.default-tenant.app.dev8.lab.iguazeng.com'
    # create a test event and invoke the function locally
    ctx = create_context()
    init_context(ctx)
    handler(ctx)



