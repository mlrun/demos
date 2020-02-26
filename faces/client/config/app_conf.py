import configparser
from utils.logger import Logger


class AppConf(object):
    def __init__(self, logger: Logger, config_path='config/init.ini'):
        self.logger = logger
        self.conf_file = config_path
        config = configparser.ConfigParser()
        config.read(self.conf_file)

        self.log_level = config['app']['log_level']
        self.nuclio_url = config['nuclio']['url']



