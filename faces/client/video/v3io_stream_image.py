import base64
import cv2
from config.app_conf import AppConf
from utils.logger import Logger
from numpy import ndarray
import numpy as np

# defining the size of images
(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class VideoFrame(object):

    def __init__(self, logger: Logger, conf: AppConf):
        self.logger = logger
        self.frame: ndarray = None
        self.gray_frame: ndarray = None
        self.gray_frame_bytes: bytes = None
        self.jpg_buffer = None
        self.conf = conf
        self.logger.info("Video frame initialize ...")

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
    def convert_to_jpg(frame):
        retval, buffer = cv2.imencode('.jpg', frame)
        if retval:
            return buffer
        else:
            raise

    @staticmethod
    def convert_frame_to_jpg_str(frame):
        #gray = VideoFrame.convert_frame_to_gray(frame)
        #resized = cv2.resize(gray,(360,480))
        jpg = VideoFrame.convert_to_jpg(frame)
        encoded = VideoFrame.b64_encode_frame(jpg)
        utf_decoded = VideoFrame.decode_as_utf(encoded)
        return utf_decoded

    @staticmethod
    def jpg_str_to_frame(frame, logger):
        jpg_original = base64.b64decode(frame)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        logger.info(img)
        return img

    @staticmethod
    def convert_frame_to_shit(frame):
        shit = 'shit'
        bytes_str = str.encode(shit)
        b64encoded_str = base64.b64encode(bytes_str)
        txt_decoded_utf = b64encoded_str.decode('utf-8')
        return txt_decoded_utf

    def find_face(self):
        faces = face_cascade.detectMultiScale(self.frame, scaleFactor=1.3,
                                              minNeighbors=4,
                                              minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = self.frame[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            return face_resize




