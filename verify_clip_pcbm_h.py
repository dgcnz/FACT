import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from training_tools.utils import test_runs
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from data import get_dataset
from models import PosthocHybridCBM, get_model
from training_tools import load_or_compute_projections, AverageMeter, MetricComputer, export


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Folder containing model/checkpoints.")
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--l2-penalty", default=0.01, type=float)
    parser.add_argument("--targets", default=[3, 6, 31, 35, 36, 37, 40, 41, \
                                             43, 46, 47, 50, 53, 64, 75, 76, 78, 80, 85, 89], \
                                             type=int, nargs='+', help="target indexes for cocostuff")
    parser.add_argument("--escfold", default=5, type=int, help="If using ESC-50 as the dataset," \
                    "you can determine the fold to use for testing.")
    parser.add_argument("--usfolds", default=[9, 10], type=int, nargs='+', help="If using US8K as the dataset," \
                    "you can determine the folds to use for testing.")

    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(',')]

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
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    
    return epoch_summary["Accuracy"]


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(),
                         "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
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
        latest_info["test_acc"] = eval_model(args, posthoc_layer, val_loader, num_classes)
        print("Final Test Accuracy:", latest_info["test_acc"])

    return latest_info


def main(args, backbone, preprocess, **kwargs):
    tar = {'target': kwargs['target']}
    train_loader, test_loader, _ , classes = get_dataset(args, preprocess, **tar)
    num_classes = len(classes)
    
    hybrid_model_path = args.pcbm_path.replace("pcbm_", "pcbm-hybrid_")
    run_info_file = Path(args.out_dir) / Path(hybrid_model_path.replace("pcbm", "run_info-pcbm")).with_suffix(".pkl").name
    
    # We use the precomputed embeddings and projections.
    train_embs, _ , train_lbls, test_embs, _ , test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=args.batch_size, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes)

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")

    return run_info


if __name__ == "__main__":
    args = config()
    metric_list = []
    og_out_dir = args.out_dir

    for i in range(len(args.seeds)):
        seed = args.seeds[i]
        # format the following path with these seeds #'artifacts/clip/cifar10_42/pcbm_cifar10__clipRN50__multimodal_concept_clipRN50_cifar10_recurse_1__lam_1e-05__alpha_0.99__seed_42.ckpt'
        args.pcbm_path = 'artifacts/clip/cifar' + args.dataset + '_' + str(seed) + '/pcbm_cifar10__clipRN50__multimodal_concept_clipRN50_cifar10_recurse_1__lam_1e-05__alpha_0.99__seed_' + str(seed) + '.ckpt'
        # Load the PCBM
        posthoc_layer = torch.load(args.pcbm_path)
        posthoc_layer = posthoc_layer.eval()
        args.backbone_name = posthoc_layer.backbone_name
        backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
        backbone = backbone.to(args.device)
        backbone.eval()

        print(f"Seed: {seed}")
        args.seed = seed
        args.out_dir = og_out_dir + "_" + str(seed)
        run_info = test_runs(args, main, concept_bank="", 
                             backbone=backbone, preprocess=preprocess, mode="vch")
        metric = run_info['test_acc']

        if isinstance(metric, (int, float)):
            print("AUC used")
            metric_list.append(metric)

        else:
            print("Accuracy used")
            metric_list.append(metric.avg)

    # export results
    out_name = "verify_clip_pcbm_h"
    export.export_to_json(out_name, metric_list)
    print("Verification results exported!")
