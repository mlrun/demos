import requests
from utils.logger import Logger
from requests.auth import HTTPBasicAuth
from config.app_conf import AppConf

content_type = 'application/json'


def generate_uri(logger: Logger, conf: AppConf):
    uri = conf.webapi_url + "/" + conf.container + "/" + conf.stream_name + "/"
    logger.debug("uri :" + uri)
    return uri


class StreamProducer(object):

    def __init__(self, logger: Logger, conf: AppConf):
        self.logger = logger
        self.conf = conf

    def create_stream(self, shards_count=100, retention_period_hours=1):
        api_host = generate_uri(self.logger, self.conf)
        headers = {"Content-Type": content_type, "X-v3io-function": "CreateStream"}

        payload = {"ShardCount": shards_count, "RetentionPeriodHours": retention_period_hours}
        response = requests.request("POST", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password), verify=False)

        self.logger.info(response)

    def send(self, record):
        api_host = generate_uri(self.logger, self.conf)
        headers = {"Content-Type": content_type, "X-v3io-function": "PutRecords"}
        payload = {"Records": [{
            "Data": record
        }
        ]}

        response = requests.request("PUT", api_host, json=payload, headers=headers,
                                    auth=HTTPBasicAuth(self.conf.username, self.conf.password),
                                    verify=False)
        self.logger.debug(response.content)
