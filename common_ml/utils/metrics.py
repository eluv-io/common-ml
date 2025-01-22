from loguru import logger
import time

class timeit:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        logger.info(f'{self.message}')
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f'Finished {self.message}...')
        logger.info(f"Elapsed time: {self.interval:.4f} seconds")