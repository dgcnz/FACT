import os
import pickle
import numpy as np
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from data import get_dataset
from models import get_model
from models.pcbm_utils_prune import UserPosthocHybridCBM
from training_tools import load_or_compute_projections, AverageMeter, MetricComputer, export


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder")
    parser.add_argument("--pcbm-path", default="", type=str, help="Trained PCBM module.")
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--num-epochs", default=20, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--l2-penalty", default=0.01, type=float)
    parser.add_argument("--print-out", default=True, type=bool)
    parser.add_argument("--token", help="Hugging Face Token", required=True)
    parser.add_argument("--shifted-class", default="", required=True)

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
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    
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
        _, predicted = torch.max(out.data, 1)
        for i in range(batch_Y.size(0)):
            label = batch_Y[i]
            correct_per_class[label] += (predicted[i] == label).item()
            total_per_class[label] += 1
    per_class_accuracy = correct_per_class / total_per_class
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    
    return epoch_summary["Accuracy"], per_class_accuracy


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes, class_idx):
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
        latest_info["test_acc"], class_accs = eval_model(args, posthoc_layer, val_loader, num_classes)
        #class_accuracies = {idx_to_class[class_idx]: acc for class_idx, acc in enumerate(class_accs)}
        latest_info["class_acc"]= class_accs[class_idx]

    return latest_info


def main(args, backbone, preprocess):

    if args.print_out == False: # For print control
        os.environ["TQDM_DISABLE"] = "1"

    torch.manual_seed(args.seed)
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    num_classes = len(classes)
    shifted_class = classes.index(args.shifted_class)
    
    hybrid_model_path = args.pcbm_path.replace("pcbm_", "pcbm-hybrid_")
    run_info_file = Path(args.out_dir) / Path(hybrid_model_path.replace("pcbm", "run_info-pcbm")).with_suffix(".pkl").name
    
    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=args.batch_size, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = UserPosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes, shifted_class)

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")
    return run_info


if __name__ == "__main__":    
    args = config()
    # Load the PCBM
    metric_list = []
    class_list = []

    # Execute main code
    #main(args, backbone, preprocess)
    for seed in args.seeds:
        args.pcbm_path = 'artifacts/outdir/pcbm_' + args.dataset + '__clip:RN50__multimodal_concept_clip:RN50_cifar100_recurse:1__lam:0.002__alpha:0.99__seed:' + str(seed) + '.ckpt'
        posthoc_layer = torch.load(args.pcbm_path)
        posthoc_layer = posthoc_layer.eval()

        # Get the backbone from the model zoo.
        args.backbone_name = posthoc_layer.backbone_name
        backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
        backbone = backbone.to(args.device)
        backbone.eval()
        print(f"Seed: {seed}")
        args.seed = seed
        
        run_info = main(args, backbone, preprocess)

        if "test_auc" in run_info:
            metric = run_info['test_auc']

        else:
            metric = run_info['test_acc']
        class_list.append(run_info['class_acc'])
        metric_list.append(metric.avg)

    out_name = "UserStudy_PCBM-h_results_"+args.dataset
    export.export_to_json(out_name, metric_list)

    print('Spurious class metrics >>>')
    print(class_list)
    print('Mean: {}'.format(np.mean(class_list)))
    print('Std: {}'.format(np.std(class_list)))
