from utils.skl.logging import SKLLogger
import numpy as np

class SKLModule(object):
    SAVE_FORMAT_EXT: str = ".skops"
    logger: SKLLogger

    def __init__(self):
        pass

    def setup(self, logger: SKLLogger):
        self.logger = logger

    def log(self, name: str, metric: float):
        assert self.logger is not None
        self.logger.log(name, metric)

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        raise NotImplementedError()

    def test(self, xs: np.ndarray, ys: np.ndarray):
        raise NotImplementedError()
    
    def dump(self, path: str):
        raise NotImplementedError()
    
    def load(self, path: str):
        raise NotImplementedError()