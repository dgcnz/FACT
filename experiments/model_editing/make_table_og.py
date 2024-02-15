"""
Runs `train_pcbm` script with different datasets, captures outputs and exports pandas dataframe with all data.
python train_pcbm.py \
    --backbone-name clip:RN50  \
    --concept-bank artifacts/outdir/broden_clip_RN50_10.0_50.pkl \
    --out-dir artifacts/outdir \
    --device cpu \
    --dataset metashift_cherrypicked_task_1_bed_cat_dog \
    --seed 10 
    --pruned-class-name bed \
    --pruned-concept-name cat \
    --lam 0.0005  \
    --sort-concepts \
    --no-use-cache
"""

import argparse
import pandas as pd
import subprocess
import logging
from pathlib import Path
import time
import ast
import yaml

OUT_DIR = "artifacts/outdir"
DATASETS = [
    "task_1_bed_cat_dog",
    "task_1_bed_dog_cat",
    "task_2_table_books_cat",
    "task_2_table_books_dog",
    "task_2_table_cat_dog",
    "task_2_table_dog_cat",
]


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone-name", type=str, choices=["clip:RN50", "resnet19_imagenet1k_v1"]
    )
    parser.add_argument("--concept-bank", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--lam", type=float)
    parser.add_argument(
        "--sort-concepts", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        choices=["cherrypicked", "seed42"],
        default="cherrypicked",
    )
    parser.add_argument(
        "--use-cache", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


class TrainPCBMScript(object):
    def __init__(self):
        pass

    def _saveio(self, logdir: Path, stdout: bytes, stderr: bytes):
        with open(logdir / "stdout.txt", "w") as f:
            f.write(stdout.decode("utf-8"))
        with open(logdir / "stderr.txt", "w") as f:
            f.write(stderr.decode("utf-8"))

    def __call__(
        self,
        backbone_name: str,
        dataset_prefix: str,
        dataset: str,
        concept_bank: str,
        device: str,
        seed: int,
        lam: float,
        sort_concepts: bool,
        use_cache: bool,
        logdir: Path,
    ):
        dataset_name = f"metashift_{dataset_prefix}_{dataset}"
        *_, pruned_class_name, pruned_concept_name, _ = dataset.split("_")

        if pruned_concept_name == "books":
            pruned_concept_name = "book"

        cmd = [
            ".venv/bin/python",
            "-m",
            "train_pcbm",
            "--backbone-name",
            backbone_name,
            "--concept-bank",
            concept_bank,
            "--out-dir",
            OUT_DIR,
            "--device",
            device,
            "--dataset",
            dataset_name,
            "--seed",
            str(seed),
            "--lam",
            str(lam),
            "--pruned-concept-name",
            pruned_concept_name,
            "--pruned-class-name",
            pruned_class_name,
        ]
        if sort_concepts:
            cmd.append("--sort-concepts")
        else:
            cmd.append("--no-sort-concepts")
        if use_cache:
            cmd.append("--use-cache")
        else:
            cmd.append("--no-use-cache")
        output = subprocess.run(cmd, capture_output=True, env={"NO_AUDIOCLIP": "1"})
        self._saveio(logdir=logdir, stdout=output.stdout, stderr=output.stderr)
        metrics_str: str = output.stdout.decode("utf-8").split("\n\n")[-1]
        return ast.literal_eval(metrics_str)


logger: logging.Logger


def setup_logging(logdir: Path, verbose: bool):
    global logger
    logging.basicConfig(
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


def get_logdir(prefix: str, seeds: list[int]):
    salt = int(time.time())  # just a random number to ensure path uniqueness
    log_dir = Path("logs") / prefix / f"{'-'.join(str(s) for s in seeds)}" / str(salt)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def make_table(
    logdir: Path,
    backbone_name: str,
    concept_bank: str,
    dataset_prefix: str,
    device: str,
    seeds: list[int],
    lam: float,
    sort_concepts: bool = True,
    use_cache: bool = False,
):
    data = []
    data_gain = []
    for seed in seeds:
        for dataset in DATASETS:
            logger.info(f"[{dataset}][{seed}] Storing logs on {logdir}")
            script = TrainPCBMScript()
            metrics = script(
                backbone_name=backbone_name,
                dataset_prefix=dataset_prefix,
                dataset=dataset,
                concept_bank=concept_bank,
                device=device,
                seed=seed,
                lam=lam,
                sort_concepts=sort_concepts,
                use_cache=use_cache,
                logdir=logdir,
            )
            logger.info(f"[{dataset}][{seed}] Run metrics {metrics}")
            base_accuracy = metrics["test_acc"] / 100
            pruned_accuracy = metrics["pruned_test_acc"] / 100
            pruned_normalize_accuracy = metrics["pruned_normalized_test_acc"] / 100
            data.append(
                {
                    "task_name": dataset,
                    "seed": seed,
                    "base_accuracy": base_accuracy,
                    "pruned_accuracy": pruned_accuracy,
                    "pruned_normalize_accuracy": pruned_normalize_accuracy,
                }
            )
            data_gain.append(
                {
                    "task_name": dataset,
                    "seed": seed,
                    "pruned_accuracy_gain": pruned_accuracy - base_accuracy,
                    "pruned_normalize_accuracy_gain": pruned_normalize_accuracy
                    - base_accuracy,
                }
            )
    log_metrics(logdir, all_metrics=data, all_metrics_gain=data_gain)


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


def log_config(args: argparse.Namespace, logdir: Path):
    with open(logdir / "args.yaml", "w") as f:
        yaml.dump(args.__dict__, f)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    logdir = get_logdir(f"og_{args.backbone_name}", args.seed)
    setup_logging(logdir, verbose=True)
    log_config(args, logdir)
    make_table(
        backbone_name=args.backbone_name,
        concept_bank=args.concept_bank,
        dataset_prefix=args.dataset_prefix,
        device=args.device,
        seeds=args.seed,
        lam=args.lam,
        sort_concepts=args.sort_concepts,
        use_cache=args.use_cache,
        logdir=logdir,
    )


if __name__ == "__main__":
    main()
