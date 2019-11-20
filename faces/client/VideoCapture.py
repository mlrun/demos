import cv2
import concurrent.futures
from config.app_conf import AppConf
from stream.image_sender import ImageSender
from utils.logger import Logger
from video.v3io_image import V3ioImage
import logging
import socket

# the number of frames to process ( set to -1 for endless stream)

INTI_FILE_PATH = "config/init-cloud.ini"
NUMBER_OF_FRAMES = -1
CAMERA_NAME = socket.gethostname()
NEW_PERSON = True

logger = Logger(level=logging.DEBUG)
app_conf = AppConf(logger, INTI_FILE_PATH)
sp = ImageSender(logger,app_conf)
cap = cv2.VideoCapture(0)
count = NUMBER_OF_FRAMES
while count > 0 or NUMBER_OF_FRAMES == -1:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        vf = V3ioImage(logger, app_conf, frame, CAMERA_NAME)
        if ret:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = {executor.submit(sp.send_image(vf,NEW_PERSON))}

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

