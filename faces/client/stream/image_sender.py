import requests
from utils.logger import Logger
from requests.auth import HTTPBasicAuth
from config.app_conf import AppConf
from video.image_abs import AbsImage

CONTENT_TYPE = 'application/json'


def generate_uri(logger: Logger, conf: AppConf):
    uri = conf.webapi_url
    logger.debug("uri :" + uri)
    return uri


def generate_nuclio_uri(logger: Logger, conf: AppConf):
    uri = conf.nuclio_url
    logger.debug("uri :" + uri)
    return uri


def generate_image_uri(logger: Logger, conf: AppConf,filename):
    file_name = filename
    uriTuple = (conf.webapi_url, conf.container,conf.username, conf.stream_name, file_name)
    uri = "/".join(uriTuple)
    logger.debug("uri :" + uri)
    return uri


def generate_file_name(current_time,is_partiitoned):
    filename_str = current_time+'.jpg'
    if is_partiitoned == "true":
        filename_str = current_time[:-4]+"/"+filename_str
    return filename_str


# http producer for sending data (needs to be unified with stream producer
class ImageSender(object):

    def __init__(self, logger: Logger, conf: AppConf):
        self.logger = logger
        self.conf = conf

    def send_image(self, img: AbsImage):
        img_data = img.convert_frame_to_bytes()
        current_time = img.get_create_time()
        file_name = generate_file_name(current_time, self.conf.partition)
        api_host = generate_image_uri(self.logger, self.conf, file_name)
        response = requests.request("PUT", api_host, data=img_data,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        self.logger.debug(response.text)
        nuclio_mount_path = self.conf.nuclio_mount
        uriTuple = (nuclio_mount_path, self.conf.stream_name, file_name)
        file_path = "/".join(uriTuple)
        payload ={"file_path":file_path, "time": current_time, "camera": "cammy"}
        self.invoke_trigger(payload)

    def invoke_trigger(self, payload):
        url = generate_nuclio_uri(self.logger, self.conf)
        headers = {"Content-Type": "application/json", "X-v3io-function": "PutRecords"}
        payload = payload
        response = requests.request("PUT", url, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        self.logger.debug(response.text)