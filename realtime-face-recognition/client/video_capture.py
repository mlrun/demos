import cv2
import concurrent.futures
from config.app_conf import AppConf
from utils.logger import Logger
from video.v3io_image import V3ioImage
import logging
import socket
import requests



def get_conf_log_level(level):
    if level == "debug":
        return logging.DEBUG
    if level == "info":
        return logging.INFO
    if level == "warn":
        return logging.WARN
    if level == "error":
        return logging.ERROR


INIT_FILE_PATH = "config/init.ini"
NUMBER_OF_FRAMES = -1
CAMERA_NAME = socket.gethostname()


app_conf = AppConf(INIT_FILE_PATH)
logger = Logger(level=get_conf_log_level(app_conf.log_level))

cap = cv2.VideoCapture(0)
count = NUMBER_OF_FRAMES
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    while count > 0 or NUMBER_OF_FRAMES == -1:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                vi = V3ioImage(logger, frame, CAMERA_NAME)
                img_json = vi.image_json
                logger.debug(img_json)
                future = {executor.submit(logger.info(requests.request("POST", app_conf.nuclio_url, json=img_json).content))}
            else:
                logger.error("read cap failed")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count = count-1
        except Exception as e:
            logger.error(e)
    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

