import sys

import nuclio_sdk
import nuclio_sdk.test
import functions.api_serving
import functions.face_prediction
import logging


def chain_call_function_mock(name, event, node=None, timeout=None, service_name_override=None):
    logger = nuclio_sdk.Logger(level=logging.DEBUG)
    logger.set_handler('default', sys.stdout, nuclio_sdk.logger.HumanReadableFormatter())
    logger.debug_with("Call function mock called", name=name, service_name_override=service_name_override)
    if name == "face_prediction":
        nuclio_plat = nuclio_sdk.test.Platform()
        return nuclio_plat.call_handler(functions.face_prediction.handler, event)
    raise RuntimeError('Call function called with unexpected function name: {0}'.format(name))


def test_api_serving():
    f = open("image.txt", "r")
    image = f.readline()
    f.close()
    nuclio_plat = nuclio_sdk.test.Platform()
    nuclio_plat._call_function_mock.side_effect = chain_call_function_mock
    event = nuclio_sdk.Event(body=image)
    nuclio_plat.call_handler(functions.api_serving.handler, event)
