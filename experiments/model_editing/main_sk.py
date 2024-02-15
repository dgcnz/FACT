from data.metashift_sk import MetaShiftSKLDataModule
from models.pcbm_sk import PCBMClassifierSKL
from utils.skl.cli import SKLCLI


def get_cli():
    cli = SKLCLI(PCBMClassifierSKL, MetaShiftSKLDataModule)
    return cli

if __name__ == "__main__":
    get_cli()


