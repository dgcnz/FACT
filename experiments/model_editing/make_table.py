from experiments.model_editing.main import get_cli
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time
import itertools
import logging
from lightning.pytorch.cli import LightningCLI
import pandas as pd

CONFIG_PATH = Path("configs/model_editing/classifier")

logger: logging.Logger

def setup_logging(verbose: bool):
    global logger
    logging.basicConfig(
        # format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.ERROR if not verbose else logging.INFO,
        handlers=[logging.FileHandler("debug.log", mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR if not verbose else logging.INFO)
    handler = logging.FileHandler("debug.log")
    logger.addHandler(handler)

class LightningScript(object):
    def __init__(self, cli_fn: callable):
        self.cli_fn = cli_fn

    def _get_config_args(self, config_paths: list[str]):
        # configs: [path1, path2, path3] -> ["-c", path1, -c path2, -c path3]
        return list(
            itertools.chain(
                *zip(
                    itertools.repeat("-c", len(config_paths)),
                    [str(path) for path in config_paths],
                )
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

    def _execute(self, args: list[str]):
        import sys

        prev_args = sys.argv
        sys.argv = [" "] + args
        ans = self.cli_fn()
        sys.argv = prev_args
        return ans

    def fit(
        self,
        task_name: str,
        config_paths: list[str],
        device: str,
        seed: int,
        ckpt_path: Path | None = None,
    ) -> LightningCLI:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, device=device, seed=seed)
        args = ["fit"] + config_args + extra_args
        if ckpt_path is not None:
            args += ["--ckpt_path", str(ckpt_path)]
        logger.info(f"RUNNING: {args}")
        return self._execute(args)

    def test(
        self,
        task_name: str,
        config_paths: list[str],
        device: str,
        seed: int,
        ckpt_path: Path,
        additional_args: list[str] = [],
    ) -> LightningCLI:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, device=device, seed=seed)
        ckpt_args = ["--ckpt_path", str(ckpt_path)]
        args = ["test"] + config_args + extra_args + ckpt_args + additional_args
        logger.info(f"RUNNING: {args}")
        return self._execute(args)


def train_and_test(base_config: Path, config_folder: Path, device: str, seed: int):
    script = LightningScript(cli_fn=get_cli)
    base_config_path = config_folder / "base.yaml"
    prune_config_path = config_folder / "prune.yaml"
    finetune_config_path = config_folder.parent / "finetune_oracle.yaml"
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
    ckpt_path = Path(run.trainer.checkpoint_callback.best_model_path)
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
    base_metrics = run.trainer.callback_metrics.copy()
    ## Test pruned
    logger.info(f"Testing {task_name} on pruned")

    run = script.test(
        task_name=task_name,
        config_paths=configs + [prune_config_path],
        device=device,
        seed=seed,
        ckpt_path=ckpt_path,
    )
    pruned_metrics = run.trainer.callback_metrics.copy()
    ## Test pruned + normalize
    logger.info(f"Testing {task_name} on pruned + normalize")

    run = script.test(
        task_name=task_name,
        config_paths=configs + [prune_config_path],
        additional_args=["--model.normalize", "True"],
        device=device,
        seed=seed,
        ckpt_path=ckpt_path,
    )

    pruned_normalize_metrics = run.trainer.callback_metrics.copy()

    logger.info(f"Finetuning {task_name}")

    run = script.fit(
        task_name=task_name,
        config_paths=configs + [finetune_config_path],
        device=device,
        seed=seed,
        ckpt_path=ckpt_path,
    )
    finetuned_ckpt_path = Path(run.trainer.checkpoint_callback.best_model_path)

    logger.info(f"Test finetuning {task_name}")

    run = script.test(
        task_name=task_name,
        config_paths=configs,
        device=device,
        seed=seed,
        ckpt_path=finetuned_ckpt_path,
    )
    finetuned_test_metrics = run.trainer.callback_metrics.copy()

    all_metrics = {"task_name": task_name}
    all_metrics_gain = {"task_name": task_name}
    for name, metrics in [
        ("base", base_metrics),
        ("pruned", pruned_metrics),
        ("pruned_normalize", pruned_normalize_metrics),
        ("finetuned", finetuned_test_metrics),
    ]:
        for k, v in metrics.items():
            # strip prefix test_ from keys and add prefix name_
            metric_name = name + "_" + k.removeprefix("test_")
            all_metrics.update({metric_name: v.item()})
            if name != "base":
                all_metrics_gain.update(
                    {f"{metric_name}_gain": v.item() - base_metrics[k].item()}
                )
    return all_metrics, all_metrics_gain


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config", type=str, default=CONFIG_PATH / "base_clip_resnet50.yaml"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
    )
    return parser


def get_all_configs():
    """Get all folders in CONFIG_PATH"""
    configs = []
    for config in CONFIG_PATH.iterdir():
        if config.is_dir():
            configs.append(config)
    return sorted(configs)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    configs = get_all_configs()
    all_metrics = []
    all_metrics_gain = []
    for seed in args.seed:
        for config_folder in configs:
            metrics, metrics_gain = train_and_test(
                base_config=args.base_config,
                config_folder=config_folder,
                device=args.device,
                seed=seed,
            )
            all_metrics.append(metrics | {"seed": seed})
            all_metrics_gain.append(metrics_gain | {"seed": seed})

    log_metrics(all_metrics, all_metrics_gain, base_config=args.base_config, seeds=args.seed)


def log_metrics(all_metrics: list[dict], all_metrics_gain: list[dict], base_config: str, seeds: list[int]):
    df = pd.DataFrame(all_metrics)
    df_gain = pd.DataFrame(all_metrics_gain)
    salt = int(time.time())
    log_dir = Path("logs") / Path(base_config).stem / f"{'-'.join(str(s) for s in seeds)}" / str(salt)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging to {log_dir}")

    df.to_csv(log_dir / "metrics.csv", index=False)
    df_gain.to_csv(log_dir / "metrics_gain.csv", index=False)

    df = df.groupby("task_name").mean().drop(columns=["seed"])
    df_gain = df_gain.groupby("task_name").mean().drop(columns=["seed"])
    df.to_csv(log_dir / "aggregated_metrics.csv", index=False)
    df_gain.to_csv(log_dir / "aggregated_metrics_gain.csv", index=False)


if __name__ == "__main__":
    main()
