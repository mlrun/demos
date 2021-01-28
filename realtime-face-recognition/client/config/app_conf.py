import configparser


class AppConf(object):
    def __init__(self, config_path='config/init.ini'):
        try:
            self.conf_file = config_path
            config = configparser.ConfigParser()
            config.read(self.conf_file)

            self.log_level = config['app']['log_level']
            self.nuclio_url = config['nuclio']['url']
        except Exception as ex:
            print(ex)


