import cv2
from config.app_conf import AppConf
from utils.logger import Logger
import io
from datetime import datetime

from video.image_abs import AbsImage


def get_current_time():
    filename = datetime.utcnow()
    # Converting filename to string in the desired format (YYYYMMDD) using strftime
    time_str = str(int(filename.strftime('%Y%m%d%H%M%S')))
    return time_str


class V3ioImage(AbsImage):

    def __init__(self, logger: Logger, conf: AppConf, frame, camera):
        self.logger = logger
        self.conf = conf
        self.frame = frame
        self.camera = camera
        self.create_time = get_current_time()
        self.logger.info("Video frame initialize ...")

    def convert_frame_to_bytes(self):
        is_success, buffer = cv2.imencode('.jpg', self.frame)
        bytes_io = io.BytesIO(buffer).getvalue()
        return bytes_io

    def get_create_time(self):
        return self.create_time






