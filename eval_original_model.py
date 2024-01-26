# This script has some functions that can be used to verify the performances of the original models

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from data import get_dataset
from models import get_model, clip_pl
from training_tools import export


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Folder containing model/checkpoints.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default=[42, 43, 44], nargs='+', type=int, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--datasets", default=['ham10k'], nargs='+', type=str)
    parser.add_argument("--eval_all", action="store_true", default=False)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=None, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=2e-3, type=float, help="learning rate")
    parser.add_argument("--max-epochs", default=20, type=int, help="Maximum number of epochs.")
    args = parser.parse_args()

    return args

@torch.no_grad()
def eval_model(args, model, loader, seed):
    model.to(args.device)
    tqdm_loader = tqdm(loader)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        
        preds = model(batch_X)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())

        current_accuracy = torch.sum(preds.argmax(dim=1) == batch_Y) / batch_Y.shape[0]
        summary_text = f"Seed - {seed} | Eval - {current_accuracy}"
        tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc

    final_accuracy = np.mean(all_preds.argmax(axis=1) == all_labels)
    
    return final_accuracy


def eval_cifar(args, seed):
    # setting the seed
    pl.seed_everything(seed, workers=True)

    _ , preprocess = get_model(args, backbone_name="clip:RN50")

    train_loader, test_loader, _ , classes = get_dataset(args, preprocess)
    num_classes = len(classes)

    print(f"Evaluating for seed: {seed}")

    # first apply linear probing and instantiate the classifier module
    finetuner = clip_pl.CLIPClassifierTrainer("RN50", n_classes=num_classes, lr=args.lr)
    trainer   = pl.Trainer(max_epochs=args.max_epochs, deterministic=True)
    trainer.fit(finetuner, train_loader)

    # then evaluate the model
    results = trainer.test(dataloaders=test_loader)
    print('Current Accuracy: ' + str(results))

    return results # average accuracy over the batches


def eval_ham(args, seed):
    model, _ , preprocess = get_model(args, backbone_name="ham10000_inception", full_model=True)
    _ , test_loader, _ , _ = get_dataset(args, preprocess)

    results = eval_model(args, model, test_loader, seed)
    print('Current AUC: ' + str(results))

    return results 


def eval_cub(args, seed):
    model, _ , preprocess = get_model(args, backbone_name="resnet18_cub", full_model=True)
    _ , test_loader, _ , _ = get_dataset(args, preprocess)

    results = eval_model(args, model, test_loader, seed)
    print('Current Accuracy: ' + str(results))

    return results 


def eval_isic(args, seed):
    model, _ , preprocess = get_model(args, backbone_name="ham10000_inception", full_model=True)
    _ , test_loader, _ , _ = get_dataset(args, preprocess)

    results = eval_model(args, model, test_loader, seed)
    print('Current AUC: ' + str(results))
    
    return results 


def eval_coco(args):
    pass


def eval_per_seed(metrics:dict, args, evaluator, seeds:list):
    for seed in seeds:
        new_args = deepcopy(args)
        new_args.seed = seed
        result = evaluator(new_args, seed)
        if args.dataset not in metrics.keys():
            metrics[args.dataset] = [result]
        else:
            metrics[args.dataset].append(result)

    print(f"Average performance per seed: {np.mean(metrics[args.dataset])}")

    return metrics

if __name__ == "__main__":
    args = config()

    # Get the backbone from the model zoo.
    metrics = {}

    if "cifar10" in args.datasets or args.eval_all:
        print("Evaluating for CIFAR10")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "cifar10"
        metrics = eval_per_seed(metrics, new_args, eval_cifar, args.seeds)
    
    if "cifar100" in args.datasets or args.eval_all:
        print("Evaluating for CIFAR10")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "cifar100"
        metrics = eval_per_seed(metrics, new_args, eval_cifar, args.seeds)
    
    if "ham10k" in args.datasets or args.eval_all:
        print("Evaluating for HAM10000")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "ham10000"
        metrics = eval_per_seed(metrics, new_args, eval_ham, args.seeds)

    if "cub" in args.datasets or args.eval_all:
        print("Evaluating for CUB")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "cub"
        metrics = eval_per_seed(metrics, new_args, eval_cub, args.seeds)
    
    if "siim_isic" in args.datasets or args.eval_all:
        print("Evaluating for SIIM_ISIC")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "siim_isic"
        metrics = eval_per_seed(metrics, new_args, eval_isic, args.seeds)

    if "coco_stuff" in args.datasets or args.eval_all:
        
        #new_args = deepcopy(args)
        #new_args.dataset = "coco_stuff"
        #metrics['coco_stuff'] = eval_coco(new_args)

        pass
    
    print("=======================\n")
    
    assert (metrics != {}), "It appears that none of the datasets you've specified are supported."

    # export results
    out_name = "evaluate_og_model.json"
    export.eval_export(out_name, metrics)
