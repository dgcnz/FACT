from lightning.pytorch.cli import LightningCLI
from models.pcbm_pl import PCBMClassifierTrainer
from data.metashift import MetaShiftDataModule
import logging


class MyLightningCLI(LightningCLI):

    def after_fit(self):
        logger = logging.getLogger()
        logger.info(f"MODELPATH={self.trainer.checkpoint_callback.best_model_path}")



def get_cli():
    cli = MyLightningCLI(PCBMClassifierTrainer, MetaShiftDataModule)
    return cli

if __name__ == "__main__":
    get_cli()
