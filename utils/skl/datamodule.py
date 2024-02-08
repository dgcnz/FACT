class SKLDataModule(object):
    is_setup: bool = False

    def prepare(self, stage: str | None = None):
        if self.is_setup:
            return
        self.setup(stage=stage)
        self.is_setup = True

    def setup(self, stage: str | None = None):
        pass

    def train_dataset(self):
        raise NotImplementedError

    def test_dataset(self):
        raise NotImplementedError
