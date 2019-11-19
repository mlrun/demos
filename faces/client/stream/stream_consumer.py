import requests
from utils.logger import Logger
from requests.auth import HTTPBasicAuth
from config.app_conf import AppConf
import json

content_type = 'application/json'


def generate_uri(logger: Logger, conf: AppConf):
    uri = conf.webapi_url + "/" + conf.container + "/" + conf.stream_name + "/"
    logger.debug("uri :" + uri)
    return uri


class StreamConsumer(object):

    def __init__(self, logger: Logger, conf: AppConf):
        self.logger = logger
        self.conf = conf

    def seek_earliest(self, shardId):
        api_host = generate_uri(self.logger, self.conf)+str(shardId)
        headers = {"Content-Type": "application/json", "X-v3io-function": "Seek"}
        payload = "{\n    \"Type\": \"EARLIEST\"\n}"

        response = requests.request("post", api_host, data=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        res = json.loads(response.text)
        loc =res["Location"]
        self.logger.debug(response.text)
        return loc

    def get_record(self, shardId, location, limit=1):
        api_host = generate_uri(self.logger, self.conf)+str(shardId)
        headers = {"Content-Type": "application/json", "X-v3io-function": "GetRecords"}
        payload = {"Location": location, "Limit": limit}
        response = requests.request("post", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        res = json.loads(response.text)
        rec = res["Records"]
        for record in rec:
            data =record["Data"]
        return data



