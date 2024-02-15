# PATH = "logs/sk_base_clip_resnet50/42-12-23-34-45-56-67-78-89-100/1708014320/metrics.csv"
import pandas as pd
import argparse
from pathlib import Path


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=Path, help="Path to logpath")
    return parser


BASE_PATH = Path("scripts/figures/out")


def print_grouped_table(grouped_df: pd.DataFrame, id: str):
    grouped = grouped_df.agg(["mean", "sem"])

    def format_stats(row: pd.Series):
        cols = [x[0] for x in row.index[::2]]
        return " & ".join(
            [f"{row[(col, 'mean')]:.3f} $\pm$ {row[(col, 'sem')]:.3f}" for col in cols]
        )

    result = grouped.apply(format_stats, axis=1)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        result.to_csv(BASE_PATH / f"{id}.csv")

def print_row(df: pd.DataFrame, id: str):
    grouped = df.agg(["mean", "sem"])
    def format_stats(row: pd.Series):
        return f'{row["mean"]:.3f} $\pm$ {row["sem"]:.3f}'

    result = grouped.apply(format_stats, axis=0)
    result = result.transpose()
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        result.to_csv(BASE_PATH / f"{id}.csv")

def main():
    parser = setup_parser()
    args = parser.parse_args()
    df_metrics = pd.read_csv(args.log_path / "metrics.csv")
    df_metrics_gain = pd.read_csv(args.log_path / "metrics_gain.csv")
    # print accuracy
    print_row(
        df_metrics.groupby(by="task_name")[
            [
                "base_accuracy",
                "pruned_accuracy",
                "pruned_normalize_accuracy",
                "finetuned_accuracy",
            ]
        ].mean(),
        id="metrics"
    )
    print_row(
        df_metrics_gain.groupby(by="task_name")[
            [
                "pruned_accuracy_gain",
                "pruned_normalize_accuracy_gain",
                "finetuned_accuracy_gain",
            ]
        ].mean(),
        id="metrics_gain"
    )
    print_grouped_table(
        df_metrics.groupby(by="task_name")[
            [
                "base_accuracy",
                "pruned_accuracy",
                "pruned_normalize_accuracy",
                "finetuned_accuracy",
            ]
        ],
        id="all_metrics"
    )
    print_grouped_table(
        df_metrics_gain.groupby(by="task_name")[
            [
                "pruned_accuracy_gain",
                "pruned_normalize_accuracy_gain",
                "finetuned_accuracy_gain",
            ]
        ],
        id="all_metrics_gain"
    )


if __name__ == "__main__":
    main()
