import cv2
from utils.logger import Logger
import io
from datetime import datetime
from video.image_abs import AbsImage
import base64
import numpy as np
import json


def get_current_time():
    filename = datetime.utcnow()
    # Converting filename to string in the desired format (YYYYMMDD) using strftime
    time_str = str(int(filename.strftime('%Y%m%d%H%M%S')))
    return time_str


class V3ioImage(AbsImage):

    def __init__(self, logger: Logger, frame, camera):
        self.logger = logger
        self.frame = frame
        self.camera = camera
        self.create_time = get_current_time()
        self.image_str = self.convert_frame_to_jpg_str()
        self.image_json = self.generate_json()

    def convert_frame_to_bytes(self):
        is_success, buffer = cv2.imencode('.jpg', self.frame)
        bytes_io = io.BytesIO(buffer).getvalue()
        return bytes_io

    def get_create_time(self):
        return self.create_time

    def generate_json(self):
        return json.dumps({"content":self.image_str, "time":self.create_time, "camera":self.camera})

    def convert_frame_to_jpg_str(self):
        #gray = self.convert_frame_to_gray(self.frame)
        #resized = cv2.resize(gray,(360,480))
        jpg = self.convert_to_jpg(self.frame)
        encoded = self.b64_encode_frame(jpg)
        utf_decoded = self.decode_as_utf(encoded)
        return utf_decoded

    @staticmethod
    def convert_frame_to_gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def b64_encode_frame(frame):
        return base64.b64encode(frame)

    @staticmethod
    def convert_gray_frame_to_bytes(frame):
        return frame.tobytes()

    @staticmethod
    def decode_frame(frame):
        return base64.b64decode(frame)

    @staticmethod
    def decode_as_utf(item):
        txt_decoded_utf = item.decode('utf-8')
        return txt_decoded_utf

    @staticmethod
    def encode_from_utf(item):
        txt_encoded_utf = item.encode('utf-8')
        return txt_encoded_utf

    @staticmethod
    def convert_to_jpg(frame):
        retval, buffer = cv2.imencode('.jpg', frame)
        if retval:
            return buffer
        else:
            raise

    def jpg_str_to_frame(self,frame):
        #utf_encoded_str =   self.encode_from_utf(frame)
        jpg_original = base64.b64decode(frame)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        return img




