import nuclio_sdk
from ..notebooks.functions.face_prediction import handler
import logging


def test_face_prediction():
    f = open("image.txt", "r")
    image = f.readline()
    f.close()
    logger = nuclio_sdk.Logger(level=logging.INFO)
    ctx = nuclio_sdk.Context(logger=logger)
    event = nuclio_sdk.Event(body=image)
    handler(ctx, event)

