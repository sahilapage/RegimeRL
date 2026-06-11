import logging

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

def get_logger(name):
    return Logger(name)

def log_training_metrics(epoch, loss, accuracy, val_loss, val_accuracy):
    logger = get_logger('training_metrics')
    logger.info(f'Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')