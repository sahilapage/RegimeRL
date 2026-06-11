import logging
import os

class Logger:
    def __init__(self, log_file=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file:
            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def log(self, message):
        self.logger.info(message)