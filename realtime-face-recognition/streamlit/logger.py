import os
import logging
from colorlog import ColoredFormatter


class Logger:
    def __init__(self, level=logging.INFO, log_file_name=None):
        self.level = level
        self.log_filename = self.set_log_file_name(log_file_name)
        self.logger = self.init_logger()

    def set_log_file_name(self, name):
        log_suffix = "{}.log"
        if name is None:
            return log_suffix.format(self.__class__.__name__)
        else:
            return log_suffix.format(name)

    def init_logger(self):
        logging.basicConfig(filename=self.log_filename, level=self.level,
                            format='%(asctime)s %(threadName)s	 %(module)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

        logger = logging.Logger(__name__,  self.level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        color_formatter = ColoredFormatter(
            "%(log_color)s %(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s")
        sh = logging.StreamHandler()
        sh.setLevel(self.level)
        sh.setFormatter(color_formatter)
        handler = logging.FileHandler(self.log_filename, mode='w')
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(sh)
        logger.info("Logger Initialized")
        logger.info('Log file location: {}'.format(os.path.abspath(self.log_filename)))
        return logger

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)