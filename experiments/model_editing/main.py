from lightning.pytorch.cli import LightningCLI
from models.pcbm_pl import PCBMClassifierTrainer
from data.metashift import MetaShiftDataModule

class MyLightningCLI(LightningCLI):
    def after_fit(self):
        print(f"MODELPATH={self.trainer.checkpoint_callback.last_model_path}")



def main():
    cli = MyLightningCLI(PCBMClassifierTrainer, MetaShiftDataModule)

if __name__ == "__main__":
    main()
