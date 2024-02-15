import argparse
import itertools
import logging
import time
from pathlib import Path

import yaml
import pandas as pd

from experiments.model_editing.main_sk import get_cli
from utils.skl.cli import SKLCLI

CONFIG_PATH = Path("configs/model_editing/classifier")

logger: logging.Logger


def setup_logging(logdir: Path, verbose: bool):
    global logger
    logging.basicConfig(
        # format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.ERROR if not verbose else logging.INFO,
        handlers=[
            logging.FileHandler(logdir / "debug.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR if not verbose else logging.INFO)
    handler = logging.FileHandler(logdir / "debug.log")
    logger.addHandler(handler)


class SKLScript(object):
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
                class_path: utils.logging.SKLTensorBoardLogger
                init_args:
                save_dir: skl_logs/
                name: task_1_bed_cat_dog
        """
        return [
            "--trainer.logger.class_path",
            "utils.skl.logging.SKLTensorBoardLogger",
            "--trainer.logger.init_args.save_dir",
            "skl_logs/",
            "--trainer.logger.init_args.name",
            f"{task_name}/seed_{seed}",
        ]

    def _get_extra_args(self, task_name: str, seed: int) -> list[str]:
        """Get seed, logger args."""
        basic_args = ["--seed_everything", str(seed)]
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
        seed: int,
        ckpt_path: Path | None = None,
    ) -> SKLCLI:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, seed=seed)
        args = config_args + extra_args
        if ckpt_path is not None:
            args += ["--ckpt_path", str(ckpt_path)]
        args += ["fit"]
        logger.info(f"RUNNING: {args}")
        return self._execute(args)

    def test(
        self,
        task_name: str,
        config_paths: list[str],
        seed: int,
        ckpt_path: Path,
        additional_args: list[str] = [],
    ) -> SKLCLI:
        config_args = self._get_config_args(config_paths=config_paths)
        extra_args = self._get_extra_args(task_name=task_name, seed=seed)
        ckpt_args = ["--ckpt_path", str(ckpt_path)]
        args = config_args + extra_args + ckpt_args + additional_args
        args += ["test"]
        logger.info(f"RUNNING: {args}")
        return self._execute(args)


def train_and_test(base_config: Path, config_folder: Path, seed: int):
    script = SKLScript(cli_fn=get_cli)
    base_config_path = config_folder / "base.yaml"
    prune_config_path = config_folder / "prune.yaml"
    # TODO: Finetune
    finetune_config_path = config_folder.parent / "sk_finetune_oracle.yaml"
    task_name = config_folder.stem

    configs = [base_config, base_config_path]
    # Train
    logger.info(f"Training {task_name}")
    run = script.fit(
        task_name=task_name,
        config_paths=configs,
        seed=seed,
    )
    ckpt_path = Path(run.trainer.best_model_path)
    # Test
    ## Test base
    logger.info(f"Testing {task_name} on base")
    run = script.test(
        task_name=task_name,
        config_paths=configs,
        seed=seed,
        ckpt_path=ckpt_path,
    )
    base_metrics = run.trainer.logger.metrics.copy()
    ## Test pruned
    logger.info(f"Testing {task_name} on pruned")

    run = script.test(
        task_name=task_name,
        config_paths=configs + [prune_config_path],
        seed=seed,
        ckpt_path=ckpt_path,
    )
    pruned_metrics = run.trainer.logger.metrics.copy()
    ## Test pruned + normalize
    logger.info(f"Testing {task_name} on pruned + normalize")

    run = script.test(
        task_name=task_name,
        config_paths=configs + [prune_config_path],
        additional_args=["--model.normalize", "True"],
        seed=seed,
        ckpt_path=ckpt_path,
    )

    pruned_normalize_metrics = run.trainer.logger.metrics.copy()

    logger.info(f"Finetuning {task_name}")

    run = script.fit(
        task_name=task_name,
        config_paths=configs + [finetune_config_path],
        seed=seed,
        ckpt_path=ckpt_path,
    )
    finetuned_ckpt_path = Path(run.trainer.best_model_path) # TODO

    logger.info(f"Test finetuning {task_name}")

    run = script.test(
        task_name=task_name,
        config_paths=configs,
        seed=seed,
        ckpt_path=finetuned_ckpt_path,
    )
    finetuned_test_metrics = run.trainer.logger.metrics.copy()

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
            all_metrics.update({metric_name: v})
            if name != "base":
                all_metrics_gain.update({f"{metric_name}_gain": v - base_metrics[k]})
    return all_metrics, all_metrics_gain


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config",
        type=str,
        default=CONFIG_PATH / "sk_base_resnet18_imagenet1k_v1.yaml",
    )
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


def get_logdir(base_config: str, seeds: list[int]):
    salt = int(time.time())
    log_dir = (
        Path("logs")
        / Path(base_config).stem
        / f"{'-'.join(str(s) for s in seeds)}"
        / str(salt)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_config(args: argparse.Namespace, logdir: Path):
    with open(logdir / "args.yaml", "w") as f:
        yaml.dump(args.__dict__, f)

    with open(logdir / "base_config.yaml", "w") as f:
        yaml.dump(yaml.safe_load(Path(args.base_config).read_text()), f)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    logdir = get_logdir(base_config=args.base_config, seeds=args.seed)
    setup_logging(logdir=logdir, verbose=args.verbose)
    log_config(args, logdir)
    configs = get_all_configs()
    all_metrics = []
    all_metrics_gain = []
    for seed in args.seed:
        for config_folder in configs:
            metrics, metrics_gain = train_and_test(
                base_config=args.base_config,
                config_folder=config_folder,
                seed=seed,
            )
            all_metrics.append(metrics | {"seed": seed})
            all_metrics_gain.append(metrics_gain | {"seed": seed})

    log_metrics(
        logdir,
        all_metrics,
        all_metrics_gain,
    )


def log_metrics(
    log_dir: Path,
    all_metrics: list[dict],
    all_metrics_gain: list[dict],
):
    df = pd.DataFrame(all_metrics)
    df_gain = pd.DataFrame(all_metrics_gain)
    logger.info(f"Logging to {log_dir}")

    df.to_csv(log_dir / "metrics.csv", index=False)
    df_gain.to_csv(log_dir / "metrics_gain.csv", index=False)

    df = df.groupby("task_name").mean().drop(columns=["seed"])
    df_gain = df_gain.groupby("task_name").mean().drop(columns=["seed"])
    df.to_csv(log_dir / "aggregated_metrics.csv", index=False)
    df_gain.to_csv(log_dir / "aggregated_metrics_gain.csv", index=False)


if __name__ == "__main__":
    main()
