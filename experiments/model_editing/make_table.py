from pathlib import Path
import re
import argparse
import subprocess
import itertools
import logging

CONFIG_PATH = Path("configs/model_editing/classifier")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config", type=str, default=CONFIG_PATH / "base_clip_resnet50.yaml"
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_all_configs():
    """Get all folders in CONFIG_PATH"""
    configs = []
    for config in CONFIG_PATH.iterdir():
        if config.is_dir():
            configs.append(config)
    return sorted(configs)


class LightningScript(object):
    entrypoint: list[str]

    def __init__(self):
        self.entrypoint = ["python", "-m", "experiments.model_editing.main"]

    def _get_config_args(self, config_paths: list[str]):
        # configs: [path1, path2, path3] -> ["-c", path1, -c path2, -c path3]
        return list(
            itertools.chain(
                *zip(itertools.repeat("-c", len(config_paths)), 
                [str(path) for path in config_paths])
            )
        )

    def _get_logger_args(self, task_name: str, seed: int) -> list[str]:
        """Get logger args.

        [Reference]
        trainer:
            logger:
                class_path: lightning.pytorch.loggers.TensorBoardLogger
                init_args:
                save_dir: lightning_logs/
                name: task_1_bed_cat_dog
        """
        return [
            "--trainer.logger.class_path",
            "lightning.pytorch.loggers.TensorBoardLogger",
            "--trainer.logger.init_args.save_dir",
            "lightning_logs/",
            "--trainer.logger.init_args.name",
            f"{task_name}/seed_{seed}",
        ]

    def _get_extra_args(self, task_name: str, device: str, seed: int) -> list[str]:
        """Get device, sed, logger args."""
        basic_args = ["--trainer.accelerator", device, "--seed_everything", str(seed)]
        logger_args = self._get_logger_args(task_name=task_name, seed=seed)
        return basic_args + logger_args
    
    def _validate_output(self, run: subprocess.CompletedProcess): 
        logger.info(f"EXIT CODE: {run.returncode}")
        if run.returncode != 0:
            logger.error(f"FAILED: {run.stderr}")
        else:
            logger.info(f"SUCCESS")


    def fit(
        self, task_name: str, config_paths: list[str], device: str, seed: int
    ) -> subprocess.CompletedProcess:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, device=device, seed=seed)
        cmd = self.entrypoint + ["fit"] + config_args + extra_args
        logger.info(f"RUNNING: {cmd}")
        run = subprocess.run(cmd, capture_output=True, text=True)
        self._validate_output(run)
        return run

    def test(
        self,
        task_name: str,
        config_paths: list[str],
        device: str,
        seed: int,
        ckpt_path: str,
    ) -> subprocess.CompletedProcess:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, device=device, seed=seed)
        ckpt_args = ["--ckpt_path", ckpt_path]
        cmd = self.entrypoint + ["test"] + config_args + extra_args + ckpt_args
        logger.info(f"RUNNING: {cmd}")
        run = subprocess.run(cmd, capture_output=True, text=True)
        self._validate_output(run)
        return run

    def _get_modelpath_from_fit_run(self, run: subprocess.CompletedProcess) -> Path:
        modelpath_pattern = re.compile(r"MODELPATH=(.*)")
        modelpath_raw: str = modelpath_pattern.search(run.stdout).group(1)
        logger.info(f"Raw captured modelpath {modelpath_raw}")
        path = Path(modelpath_raw).readlink()
        logger.info(f"Casted modelpath {path}")
        return path

    def _get_test_accuracy_from_test_run(
        self, run: subprocess.CompletedProcess
    ) -> float:
        test_accuracy_pattern = re.compile(r"test_accuracy.*(\d+\.\d+)")
        test_accuracy_raw: str = test_accuracy_pattern.search(run.stdout).group(1)
        logger.info(f"Raw captured test_accuracy {test_accuracy_raw}")
        test_accuracy = float(test_accuracy_raw)
        logger.info(f"Casted test_accuracy {test_accuracy}")
        return test_accuracy


def train_and_test_2(base_config: Path, config_folder: Path, device: str, seed: int):
    script = LightningScript()
    base_config_path = config_folder / "base.yaml"
    prune_config_path = config_folder / "prune.yaml"
    task_name = config_folder.stem

    configs = [base_config, base_config_path]
    # Train
    logger.info(f"Training {task_name}")
    run = script.fit(
        task_name=task_name,
        config_paths=configs,
        device=device,
        seed=seed,
    )
    ckpt_path = script._get_modelpath_from_fit_run(run=run)
    # Test
    ## Test base
    logger.info(f"Testing {task_name} on base")
    run = script.test(
        task_name=task_name,
        config_paths=configs,
        device=device,
        seed=seed,
        ckpt_path=ckpt_path,
    )
    base_test_accuracy = script._get_test_accuracy_from_test_run(run=run)
    ## Test pruned
    logger.info(f"Testing {task_name} on pruned")

    run = script.test(
        task_name=task_name,
        config_paths=configs + [prune_config_path],
        device=device,
        seed=seed,
        ckpt_path=ckpt_path,
    )
    pruned_test_accuracy = script._get_test_accuracy_from_test_run(run=run)
    logger.info(f"Base test accuracy: {base_test_accuracy}")
    logger.info(f"Pruned test accuracy: {pruned_test_accuracy}")
    logger.info(f"Accuracy improvement: {pruned_test_accuracy - base_test_accuracy}")
    return base_test_accuracy, pruned_test_accuracy


def train_and_test(base_config: Path, config_folder: Path, device: str, seed: int):
    base_config_path = config_folder / "base.yaml"
    prune_config_path = config_folder / "prune.yaml"
    task_name = config_folder.stem

    entrypoint = ["python", "-m", "experiments.model_editing.main"]
    basic_args = ["--trainer.accelerator", device, "-c", base_config]
    command = entrypoint + ["fit"] + basic_args + ["-c", base_config_path]
    print("Training")
    run = subprocess.run(command, capture_output=True, text=True)
    modelpath_pattern = re.compile(r"MODELPATH=(.*)")
    test_accuracy_pattern = re.compile(r"test_accuracy.*(\d+\.\d+)")

    path = Path(modelpath_pattern.search(run.stdout).group(1)).readlink()
    print("Running baseline")
    command = (
        entrypoint
        + ["test"]
        + basic_args
        + ["-c", base_config_path]
        + ["--ckpt_path", path]
    )
    run = subprocess.run(command, capture_output=True, text=True)
    test_accuracy_base = test_accuracy_pattern.search(run.stdout).group(1)
    test_accuracy_base = float(test_accuracy_base)
    print("Running prune")

    command = (
        entrypoint
        + ["test"]
        + basic_args
        + ["-c", base_config_path, "-c", prune_config_path]
        + ["--ckpt_path", path]
    )
    run = subprocess.run(command, capture_output=True, text=True)
    test_accuracy_prune = test_accuracy_pattern.search(run.stdout).group(1)
    test_accuracy_prune = float(test_accuracy_prune)
    print(f"ACC IMPROVEMENT {test_accuracy_base} -> {test_accuracy_prune}")


def main():
    parser = setup_parser()
    args = parser.parse_args()
    configs = get_all_configs()
    for config_folder in configs:
        train_and_test_2(
            base_config=args.base_config,
            config_folder=config_folder,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
