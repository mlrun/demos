from functions.read_stocks import handler, init_context
from nuclio_sdk import Context, Logger
import os


# create nuclio empty context for testing
def create_context():
    logger = Logger(level=20)
    ctx = Context(logger=logger)
    return ctx


def test_read_stocks():
    os.environ['V3IO_FRAMESD'] = 'https://framesd.default-tenant.app.app-lab-3-0-1-azure.iguazio-cd2.com'
    os.environ['TOKEN'] = '35794f28-b05a-488f-8f40-ae6d31d87949'
    # create a test event and invoke the function locally
    ctx = create_context()
    init_context(ctx)
    #event = nuclio.Event(body='')
    event = None
    handler(ctx, event)
