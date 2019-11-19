from stream.stream_abs import AbsStream
import requests
from utils.logger import Logger
from requests.auth import HTTPBasicAuth
from config.app_conf import AppConf
from ast import literal_eval


SHARDS_COUNT = 3
RETENTION_PERIOD_HOURS = 1


def generate_uri(logger: Logger, conf: AppConf, shard: str = None):
    uri = conf.webapi_url + "/" + conf.container + "/" + conf.stream_name + "/"
    if shard is not None:
        uri += "/"+shard
    logger.debug("uri :" + uri)
    return uri


class V3ioStream(AbsStream):
    def __init__(self, logger: Logger, conf: AppConf):
        self.logger = logger
        self.conf = conf

    # creating v3io stream
    def create_stream(self):
        api_host = generate_uri(self.logger, self.conf)
        headers = {"Content-Type": "application/json", "X-v3io-function": "CreateStream"}
        payload = {"ShardCount": SHARDS_COUNT, "RetentionPeriodHours": RETENTION_PERIOD_HOURS}
        response = requests.request("POST", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)

        self.logger.info(response)

    def put_item(self, item):
        api_host = generate_uri(self.logger, self.conf)
        headers = {"Content-Type": "application/json", "X-v3io-function": "PutRecords"}
        payload = {"Records": [{
            "Data": item
        }
        ]}
        response = requests.request("PUT", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        self.logger.debug(response.text)

    def get_item(self, item):
        pass

    def get_location(self, shard):
        api_host = generate_uri(self.logger, self.conf, str(shard))
        payload = {
            "Type": "EARLIEST"
        }

        headers = {
            "Content-Type": "application/json",
            "X-v3io-function": "Seek",
            # "<Authorization OR X-v3io-session-key>": "<value>"
        }

        response = requests.request("POST", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)
        self.logger.debug(response.content.decode('utf8'))
        data = literal_eval(response.content.decode('utf8'))

        return data

    def get_records(self, shard):
        loc = self.get_location(shard)
        loc_val = loc['Location']
        api_host = generate_uri(self.logger, self.conf, str(shard))
        headers = {
            "Content-Type": "application/json",
            "X-v3io-function": "GetRecords",
            # "<Authorization OR X-v3io-session-key>": "<value>"
        }
        payload = {"Location": loc_val, "Limit": 1}

        response = requests.request("POST", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)
        #dict = response.json()
        #return response.content
        return response

    def delete_stream(self):
        # delete shards
        for x in range(SHARDS_COUNT):
            api_host = generate_uri(self.logger, self.conf)
            api_host += "/"+str(x)
            self.logger.info(api_host)
            response = requests.delete(api_host,auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)
            self.logger.info(response)
        # delete stream path
        api_host = generate_uri(self.logger, self.conf)
        response = requests.delete(api_host, auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)
        self.logger.info(response)


