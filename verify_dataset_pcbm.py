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
import pickle
import numpy as np
import torch
from re import sub
from training_tools.utils import test_runs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections, export
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import average_precision_score


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Folder containing model/checkpoints.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--targets", default=[3, 6, 31, 35, 36, 37, 40, 41, \
                                             43, 46, 47, 50, 53, 64, 75, 76, 78, 80, 85, 89], \
                                             type=int, nargs='+', help="target indexes for cocostuff")
    parser.add_argument("--escfold", default=5, type=int, help="If using ESC-50 as the dataset," \
                    "you can determine the fold to use for testing.")
    parser.add_argument("--usfolds", default=[9, 10], type=int, nargs='+', help="If using US8K as the dataset," \
                    "you can determine the folds to use for testing.")
    
    parser.add_argument("--random_proj", action="store_true", default=False, help="Whether to use random projection matrix")

    parser.add_argument("--identity_proj", action="store_true", default=False, help="Whether to use identity projection matrix")
    
                    

    #if one of the tree parameters below is set to None a grid search will be performed 
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=None, type=float, help="Regularization strength.")

    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(',')]
    return args


def run_linear_probe(args, train_data, test_data):
    print("START LINEAR PROBE...")
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    print(set(train_labels))
    print(len(train_features), len(train_labels))
    

    if args.lam is None:
        #Get the best possible alpha (args.lam) using the validation set 
        # Define the parameter grid for grid search
        train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, train_size= 0.8, stratify=None, random_state=args.seed)
        param_grid = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

        best_score = -float('inf')
        best_lam = None

        # Perform grid search
        for param in param_grid:
            classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                                alpha=param, l1_ratio=args.alpha, verbose=0,
                                penalty="elasticnet", max_iter=5000)
            classifier.fit(train_features, train_labels)
            
            # Evaluate on the validation set
            y_pred = classifier.predict(val_features)
            if test_labels.max() == 1:
                score = roc_auc_score(val_labels, y_pred)
            else:
                score = accuracy_score(val_labels, y_pred)
            
            # Update best parameters if current configuration is better
            if score > best_score:
                best_score = score
                best_lam = param

        print(best_lam)
        print(best_score)
    else:
        best_lam = args.lam

    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=best_lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=10000) # TODO: change to OLS package function such that I can do tests and stuff on it. essentially a logistic regression. 
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # If it's a binary task, we compute auc
    if args.dataset == 'coco_stuff':
      run_info["test_auc"] = average_precision_score(test_labels, classifier.decision_function(test_features))
      run_info["train_auc"] = average_precision_score(train_labels, classifier.decision_function(train_features))
    
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))

    return run_info, classifier.coef_, classifier.intercept_


def main(args, concept_bank, backbone, preprocess, **kwargs):
    tar = {'target': kwargs['target']} if ('target' in kwargs.keys()) else {'target': 3}
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess, **tar)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)

    shape = concept_bank.vectors.shape

    if args.random_proj:
        concept_bank.vectors = None
        concept_bank.intercepts = None
        concept_bank.norms = None
        concept_bank.margin_info = None
        print(concept_bank.vectors)

        concept_bank.vectors = torch.randn((shape[0], shape[1])).to(args.device)
        print(concept_bank.vectors)
        concept_bank.norms = torch.norm(concept_bank.vectors, p=2, dim=1, keepdim=True).detach()
        print(concept_bank.norms.shape)
        concept_bank.vectors /= concept_bank.norms
        concept_bank.norms = torch.norm(concept_bank.vectors, p=2, dim=1, keepdim=True).detach()
        concept_bank.intercepts = torch.zeros(shape[0],1).to(args.device)

    elif args.identity_proj:
        concept_bank.vectors = None
        concept_bank.intercepts = None
        concept_bank.norms = None
        concept_bank.margin_info = None
        print('identity projection used')
        concept_bank.vectors = torch.eye(n=shape[1]).to(args.device) #(embedding dim x embedding dim identity matrix)
        concept_bank.norms = torch.norm(concept_bank.vectors, p=2, dim=1, keepdim=True).detach()

        concept_bank.intercepts = torch.zeros(shape[0],1).to(args.device)

    
    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    _ , train_projs, train_lbls, _ , test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    model_id = f"{args.dataset}_{args.backbone_name}{conceptbank_source}lam{args.lam}_alpha{args.alpha}_seed{args.seed}"
    model_id = f"{model_id}target{kwargs['target']}" if (args.dataset == "coco_stuff") else model_id
    model_path = os.path.join(args.out_dir, f"pcbm_{model_id}.ckpt")
    model_path = sub(":", "", model_path)
    torch.save(posthoc_layer, model_path)

    run_info_file = os.path.join(args.out_dir, f"run_info-pcbm_{model_id}.pkl")
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    if num_classes > 1:
        # Prints the Top-5 Concept Weigths for each class.
        print(posthoc_layer.analyze_classifier(k=5))
        print(posthoc_layer.analyze_classifier(k=5, print_lows=True))

    print(f"Model saved to : {model_path}")
    print(run_info)

    return run_info

if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    metric_list = []
    og_out_dir = args.out_dir

    for seed in args.seeds:
        print(f"Seed: {seed}")
        args.seed = seed
        args.out_dir = og_out_dir 
        run_info = test_runs(args, main, concept_bank, backbone, preprocess, mode="vdr")

        if "test_auc" in run_info:
            print("AUC used")
            metric = run_info['test_auc']

        else:
            print("Accuracy used")
            metric = run_info['test_acc']

        metric_list.append(metric)
    
    # export results
    out_name = "verify_dataset_pcbm"
    export.export_to_json(out_name, metric_list)
    print("Verification results exported!")
    