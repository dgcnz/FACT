import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from re import sub
from training_tools.utils import test_runs
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from data import get_dataset
from models import PosthocHybridCBM, get_model
from sklearn.metrics import average_precision_score
from training_tools import (
    load_or_compute_projections,
    AverageMeter,
    MetricComputer,
    export,
)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Folder containing model/checkpoints.",
    )
    parser.add_argument(
        "--concept-bank", required=True, type=str, help="Path to the concept bank."
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--seeds", default="42", type=str, help="Random seeds")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)

    parser.add_argument(
        "--alpha",
        default=0.99,
        type=float,
        help="Sparsity coefficient for elastic net.",
    )
    parser.add_argument(
        "--lam", default=1e-5, type=float, help="Regularization strength."
    )
    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--l2-penalty", default=0.01, type=float)
    parser.add_argument(
        "--targets",
        default=[
            3,
            6,
            31,
            35,
            36,
            37,
            40,
            41,
            43,
            46,
            47,
            50,
            53,
            64,
            75,
            76,
            78,
            80,
            85,
            89,
        ],
        type=int,
        nargs="+",
        help="target indexes for cocostuff",
    )
    parser.add_argument(
        "--escfold",
        default=5,
        type=int,
        help="If using ESC-50 as the dataset,"
        "you can determine the fold to use for testing.",
    )
    parser.add_argument(
        "--usfolds",
        default=[9, 10],
        type=int,
        nargs="+",
        help="If using US8K as the dataset,"
        "you can determine the folds to use for testing.",
    )

    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(",")]

    return args


@torch.no_grad()
def eval_model(args, posthoc_layer, loader, num_classes):
    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []

    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
        out = posthoc_layer(batch_X)
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y)
        epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        summary_text = "Eval - " + " ".join(summary_text)
        tqdm_loader.set_description(summary_text)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if args.dataset == "coco_stuff":
        AP = average_precision_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return AP

    elif all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])

        return auc

    return epoch_summary["Accuracy"]


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(), "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = (
                cls_loss
                + args.l2_penalty * (posthoc_layer.residual_classifier.weight**2).mean()
            )
            loss.backward()
            optimizer.step()

            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
            metrics = computer(out, batch_Y)
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = " ".join(summary_text)
            tqdm_loader.set_description(summary_text)

        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = args
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(
            args, posthoc_layer, val_loader, num_classes
        )
        print("Final Test Accuracy:", latest_info["test_acc"])

    return latest_info


def main(args, backbone, preprocess, posthoc_layer, **kwargs):
    tar = {"target": kwargs["target"]} if ("target" in kwargs.keys()) else {"target": 3}
    train_loader, test_loader, _, classes = get_dataset(args, preprocess, **tar)
    num_classes = len(classes)

    hybrid_model_path = args.pcbm_path.replace("pcbm_", "pcbm-hybrid_")
    hybrid_model_path = sub(":", "", hybrid_model_path)
    hybrid_model_path = sub(
        "target_[0-9]+", "target_" + str(kwargs["target"]), hybrid_model_path
    )  # now we only have to input one file destination as a general form
    run_info_file = (
        Path(args.out_dir)
        / Path(hybrid_model_path.replace("pcbm", "run_info-pcbm"))
        .with_suffix(".pkl")
        .name
    )

    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections(
        args, backbone, posthoc_layer, train_loader, test_loader
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)

    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(
        hybrid_model.residual_classifier.parameters(), lr=args.lr
    )
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()

    # Train PCBM-h
    run_info = train_hybrid(
        args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes
    )

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    print(f"Saved to {hybrid_model_path}, {run_info_file}.")

    return run_info


if __name__ == "__main__":
    args = config()
    metric_list = []
    og_out_dir = args.out_dir

    for i in range(len(args.seeds)):
        seed = args.seeds[i]
        args.seed = seed

        # Load the PCBM
        conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0]
        args.pcbm_path = (
            "artifacts/coco-stuff/" if (args.dataset == "coco-stuff") else "artifacts/"
        )
        args.pcbm_path += f"pcbm_{args.dataset}_{args.backbone_name}{conceptbank_source}lam{args.lam}_alpha{args.alpha}_seed{args.seed}.ckpt"
        if ":" in args.pcbm_path:
            args.pcbm_path = sub(":", "", args.pcbm_path)

        if args.dataset == "coco_stuff":
            args.out_dir = og_out_dir
            run_info = test_runs(
                args,
                main,
                concept_bank="",
                backbone=None,
                preprocess=None,
                mode="vdh",
                get_model=get_model,
            )
        else:
            posthoc_layer = torch.load(args.pcbm_path)
            args.backbone_name = posthoc_layer.backbone_name
            posthoc_layer.eval()
            backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
            backbone = backbone.to(args.device)
            backbone.eval()
            print(f"Seed: {seed}")
            args.out_dir = og_out_dir
            run_info = test_runs(
                args,
                main,
                concept_bank="",
                backbone=backbone,
                preprocess=preprocess,
                mode="vdh",
            )

        metric = run_info["test_acc"]

        if isinstance(metric, (int, float)):
            print("AUC used")
            metric_list.append(metric)

        else:
            print("Accuracy used")
            metric_list.append(metric.avg)

    # export results
    out_name = "verify_dataset_pcbm_h"
    export.export_to_json(out_name, metric_list)
    print("Verification results exported!")
