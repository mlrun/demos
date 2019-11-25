import configparser
from utils.logger import Logger


class AppConf(object):
    def __init__(self, logger: Logger, config_path='config/init.ini'):
        self.logger = logger
        self.conf_file = config_path
        config = configparser.ConfigParser()
        config.read(self.conf_file)

        self.log_level = config['app']['log_level']
        self.partition = config['app']['partition']

        self.webapi_url = config['webapi']['url']
        self.container = config['webapi']['container']
        self.stream_name = config['webapi']['stream_name']
        self.data_set_path = config['webapi']['dataset_path']
        self.username = config['auth']['username']
        self.password = config['auth']['password']
        self.session_key = config['auth']['session_key']
        self.nuclio_mount = config['nuclio']['mount']
        self.nuclio_url = config['nuclio']['url']



