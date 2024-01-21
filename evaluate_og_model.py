# This script has some functions that can be used to verify the results of PCBM with clip features
# First run the comand below 
# python learn_concepts_multimodal.py --backbone-name="clip:RN50" --classes=cifar10 --out-dir="artifacts/multimodal" --recurse=1
# This script has only the "recurse" hyperparameter which we leave at its default value of 1.

# The code will run a gridsearch over the hyperparameters of the method. In particular:
# 1) lr
# 2) lam
# 3) alpha

import argparse
import os
import numpy as np
import torch
import clip

from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.special import softmax
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections, export
from torch.utils.data import DataLoader, random_split
from training_tools import load_or_compute_projections, AverageMeter, MetricComputer


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    parser.add_argument("--datasets", default=None, nargs='+', type=str)
    parser.add_argument("--eval_all", action="store_true", default=False)

    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(',')]
    return args

@torch.no_grad()
def eval_model(args, model, loader, num_classes, clip=False, text_inputs=None):
    #if clip == True and text_inputs is None:
    #    raise ValueError("Must pass text inputs if clip=False")

    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        
        if clip:
            out = get_clip_output(args, model, batch_X, text_inputs)
        else:
            out = model(batch_X)

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

@torch.no_grad()
def get_clip_output(args, model, batch_X, text_inputs):

    # Calculate features
    image_features = model.encode_image(batch_X)
    text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    return indices
    

@torch.no_grad()
def cifar10(args):
    backbone, preprocess = get_model(args, backbone_name="clip:RN50", full_model=False)
    backbone = backbone.to(args.device)
    backbone.eval()

    _, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    num_classes = len(classes)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(args.device)

    results = eval_model(args, backbone, test_loader, num_classes, text_inputs)

    return results.avg #average accuracy over the batches


@torch.no_grad()
def cifar100():
    pass

@torch.no_grad()
def ham10k():
    pass

@torch.no_grad()
def cub():
    pass

@torch.no_grad()
def isic():
    pass

@torch.no_grad()
def coco_stuff():
    pass


if __name__ == "__main__":
    args = config()

    # Get the backbone from the model zoo.

    metrics = {}

    if "cifar10" in args.datasets or args.eval_all:
        new_args = deepcopy(args)
        new_args.dataset = "cifar10"
        metrics['cifar10'] = cifar10(new_args)
    
    if "cifar100" in args.datasets or args.eval_all:
        new_args = deepcopy(args)
        new_args.dataset = "cifar100"
        metrics['cifar100'] = cifar100(new_args)
    
    if "ham10k" in args.datasets or args.eval_all:
        new_args = deepcopy(args)
        new_args.dataset = "ham10000"
        metrics['ham10k'] = ham10k(new_args)

    if "cub" in args.datasets or args.eval_all:
        new_args = deepcopy(args)
        new_args.dataset = "cub"
        metrics['cub'] = cub(new_args)  
    
    if "isic" in args.datasets or args.eval_all:
        new_args = deepcopy(args)
        new_args.dataset = "isic"
        metrics['isic'] = isic(new_args)

    if "coco_stuff" in args.datasets or args.eval_all:
        
        #new_args = deepcopy(args)
        #new_args.dataset = "coco_stuff"
        #metrics['coco_stuff'] = coco_stuff()
        pass

    print(metrics)

    # export results
    out_name = "evaluate_og_model.json"
    out_name = os.path.join(args.out_dir, out_name)
    export.export_to_json(out_name, metrics)