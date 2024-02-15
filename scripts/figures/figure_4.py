import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import yaml
from enum import Enum
from collections import namedtuple


class Implementation(Enum):
    OG = "OG"
    SK = "SK"
    PL = "PL"

    def __str__(self):
        return self.value


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        type=Implementation,
        choices=list(Implementation),
        help="Which implementation to use: OG (original), SK (sklearn), PL (pytorch-lightning",
    )
    parser.add_argument("--experiment_paths", nargs="+", type=Path)
    return parser


Hyperparameters = namedtuple("Hyperparameters", "lam")


N_CONCEPTS=170
N_CLASSES=5
PLOT_PATH=Path("scripts/figures/out")
#FONT = "Californian FB"
# TITLE_KWARGS = {'fontname': FONT, 'weight': 'bold', 'fontsize':14}
# PLOT_KWARGS = {'fontname': FONT, 'fontsize': 12}
TITLE_KWARGS = {}
PLOT_KWARGS = {}

def get_hyperparameters(
    experiment_path: Path, implementation: Implementation
) -> Hyperparameters:
    if implementation in [Implementation.SK, Implementation.PL]:
        with open(experiment_path / "base_config.yaml") as f:
            config = yaml.safe_load(f)
            return Hyperparameters(lam=config["model"]["lam"])
    elif implementation == Implementation.OG:
        with open(experiment_path / "args.yaml") as f:
            args = yaml.safe_load(f)
            return Hyperparameters(lam=round(args["lam"] * N_CONCEPTS * N_CLASSES, 3))
    else:
        raise NotImplementedError("Unknown implementation")


def make_joint_df(experiment_paths: list[Path], implementation: Implementation):
    dfs: list[pd.DataFrame] = []
    for experiment_path in experiment_paths:
        hps = get_hyperparameters(experiment_path, implementation)
        df = pd.read_csv(experiment_path / "aggregated_metrics_gain.csv")
        df["lam"] = hps.lam
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def make_plot(experiment_paths: list[Path], implementation: Implementation):
    df = make_joint_df(experiment_paths=experiment_paths, implementation=implementation)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    df.boxplot("pruned_accuracy_gain", "lam", ax=axes[0], showmeans=True)
    df.boxplot("pruned_normalize_accuracy_gain", "lam", ax=axes[1], showmeans=True)
    axes[0].set_ylim(-0.02, 0.04)
    axes[1].set_ylim(-0.02, 0.04)
    axes[0].set_title("Prune", **TITLE_KWARGS)
    axes[0].set_ylabel("Accuracy gain",**PLOT_KWARGS)
    axes[0].set_xlabel("$\lambda$", **PLOT_KWARGS)
    axes[1].set_xlabel("$\lambda$", **PLOT_KWARGS)
    axes[1].set_title("Prune+Normalize", **TITLE_KWARGS)
    fig.tight_layout()
    fig.suptitle("")
    fig.savefig(PLOT_PATH / f"{implementation.value}_combinations.jpg", dpi=900)


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    parser = setup_parser()
    args = parser.parse_args()
    make_plot(
        experiment_paths=args.experiment_paths, implementation=args.implementation
    )


if __name__ == "__main__":
    main()
