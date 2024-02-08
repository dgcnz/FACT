from typing import Type

from jsonargparse import ActionConfigFile, ArgumentParser
from lightning import seed_everything

from models.pcbm_sk import SKLModule
from utils.skl.datamodule import SKLDataModule
from utils.skl.trainer import SKLTrainer


class SKLCLI(object):
    model: SKLModule
    datamodule: SKLDataModule
    trainer: SKLTrainer

    def __init__(
        self, model_class: Type[SKLModule], datamodule_class: Type[SKLDataModule]
    ):
        self.parser = ArgumentParser()
        self.parser.add_class_arguments(model_class, "model")
        self.parser.add_class_arguments(datamodule_class, "data")
        self.parser.add_class_arguments(SKLTrainer, "trainer")
        self.parser.add_argument("--seed_everything", default=42, type=int)
        self.parser.add_argument("--ckpt_path", type=str)  
        self.parser.add_argument("-c", "--config", action=ActionConfigFile)

        subcommands = self.parser.add_subcommands()
        subcommands.add_subcommand("fit", self._register_method("fit"))
        subcommands.add_subcommand("test", self._register_method("test"))
        subcommands.add_subcommand("fit_test", self._register_method("fit_test"))
        subcommands.add_subcommand(
            "fit_prune_test", self._register_method("fit_prune_test")
        )
        subcommands.add_subcommand(
            "fit_prune_normalize_test",
            self._register_method("fit_prune_normalize_test"),
        )
        # add subcommand for all methods

        self.parser.link_arguments("seed_everything", "model.seed", apply_on="parse")
        args = self.parser.parse_args()
        seed_everything(args.seed_everything)
        args = self.parser.instantiate_classes(args)
        self.model: SKLModule = args.model
        self.datamodule: SKLDataModule = args.data
        self.trainer: SKLTrainer = args.trainer
        self.ckpt_path = args.ckpt_path
        # call subcommand
        self.__getattribute__(args.subcommand)(**args[args.subcommand])

    def _register_method(self, method: str):
        parser = ArgumentParser()
        parser.add_method_arguments(self.__class__, method)
        return parser

    def fit(self):
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)

    def test(self):
        self.trainer.test(self.model, self.datamodule, ckpt_path=self.ckpt_path)

    def fit_test(self):
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        self.trainer.test(self.model, self.datamodule, ckpt_path=self.ckpt_path)

    def fit_prune_test(self, pruned_concept_class_pairs: list[tuple[int, int]]):
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        self.model.prune(concept_class_pairs=pruned_concept_class_pairs)
        self.trainer.test(self.model, self.datamodule, ckpt_path=self.ckpt_path)

    def fit_prune_normalize_test(
        self, pruned_concept_class_pairs: list[tuple[int, int]]
    ):
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        self.model.prune_and_normalize(concept_class_pairs=pruned_concept_class_pairs)
        self.trainer.test(self.model, self.datamodule, ckpt_path=self.ckpt_path)