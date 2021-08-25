import tensorflow as tf
import signal
import sys

class Signals:
    def __init__(self, logger):
        self.logger = logger
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.logger.info('Disposing resources....')
        self.logger.info('Gracefully Stopping Application!')
        sys.exit()
