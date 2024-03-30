from data.metashift import (
    MetaShiftDataModule,
    NNProjector,
    PreprocessorEnum,
)
from utils.skl.datamodule import SKLDataModule


class MetaShiftSKLDataModule(SKLDataModule):
    TRAIN_SIZE: int = 500
    TEST_SIZE: int = 500

    def __init__(
        self,
        task_name: str,
        projector: NNProjector,
        preprocessor_name: PreprocessorEnum,
        train_on_test: bool = False,
        holdout_size: float | None = None,
    ):
        self._datamodule = MetaShiftDataModule(
            task_name=task_name,
            projector=projector,
            preprocessor_name=preprocessor_name,
            train_batch_size=self.TRAIN_SIZE,
            test_batch_size=self.TEST_SIZE,
            train_on_test=train_on_test,
            holdout_size=holdout_size,
        )

    def setup(self, stage=None):
        self._datamodule.setup(stage=stage)

    def train_dataset(self):
        dataloader = list(self._datamodule.train_dataloader())
        assert len(dataloader) == 1, "Only one batch supported"
        return dataloader[0]["input"], dataloader[0]["label"]

    def test_dataset(self):
        dataloader = list(self._datamodule.test_dataloader())
        assert len(dataloader) == 1, "Only one batch supported"
        return dataloader[0]["input"], dataloader[0]["label"]
