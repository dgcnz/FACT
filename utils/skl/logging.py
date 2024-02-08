import time
from pathlib import Path

from tensorboardX import SummaryWriter


class SKLLogger(object):
    DEFAULT_LOGDIR: Path = Path("skl_logs")
    logdir: Path
    metrics: dict[str, float]


    def __init__(self, save_dir: str | None, name: str | None):
        timestamp = str(int(time.time()))
        self.save_dir = Path(save_dir) or self.DEFAULT_LOGDIR
        self.name = name + "/" + timestamp if name else timestamp
        self.logdir = self.save_dir / self.name

    def log(self, name: str, metric: float):
        raise NotImplementedError()

    def open(self):
        return self

    def close(self):
        pass


class SKLTensorBoardLogger(SKLLogger):
    metrics: dict[str, float] = {}
    tb_writer: SummaryWriter

    def __init__(self, save_dir: str | None = None, name: str | None = None):
        super().__init__(save_dir, name)
        self.open()

    def log(self, name: str, metric: float):
        # No global step is needed since skl algorithms run in just one batch
        self.tb_writer.add_scalar(name, metric)
        self.metrics[name] = metric

    def open(self):
        self.tb_writer = SummaryWriter(logdir=self.logdir)
        self.metrics = {}
        return self

    def close(self):
        self.tb_writer.close()
        self.metrics = {}
        self.tb_writer = None
