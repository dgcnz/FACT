from utils.skl.logging import SKLLogger, SKLTensorBoardLogger
import pprint
from utils.skl.module import SKLModule
from utils.skl.datamodule import SKLDataModule


class SKLTrainer(object):
    logger: SKLLogger
    best_model_path: str | None = None

    def __init__(self, logger: SKLLogger | None = None):
        self.logger = logger or SKLTensorBoardLogger()

    def _setup(self, model: SKLModule, datamodule: SKLDataModule, ckpt_path: str | None = None):
        model.setup(logger=self.logger.open())
        datamodule.prepare()
        if ckpt_path is not None:
            model.load(ckpt_path)

    def _get_model_path(self, model: SKLModule):
        return (self.logger.logdir / "model").with_suffix(model.SAVE_FORMAT_EXT)


    def fit(self, model: SKLModule, datamodule: SKLDataModule, ckpt_path: str | None = None):
        self._setup(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        inputs, labels = datamodule.train_dataset()
        model.fit(inputs, labels)
        self.best_model_path = self._get_model_path(model)
        model.dump(self.best_model_path)
        pprint.pprint(self.logger.metrics, indent=4)


    def test(self, model: SKLModule, datamodule: SKLDataModule, ckpt_path: str | None = None):
        self._setup(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        inputs, labels = datamodule.test_dataset()
        model.test(inputs, labels)
        pprint.pprint(self.logger.metrics, indent=4)
