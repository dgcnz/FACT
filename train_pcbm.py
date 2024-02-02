import argparse
import os
import pickle
import numpy as np
import torch
from re import sub
from training_tools.utils import train_runs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    # For the above: Please make sure to output the COCO-Stuff results in "outdir/coco-stuff"
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--print-out', action="store_true", default=True)
    parser.add_argument("--targets", default=[3, 6, 31, 35, 36, 37, 40, 41, \
                                             43, 46, 47, 50, 53, 64, 75, 76, 78, 80, 85, 89], \
                                             type=int, nargs='+', help="target indexes for cocostuff")
    parser.add_argument("--escfold", default=5, type=int, help="If using ESC-50 as the dataset," \
                    "you can determine the fold to use for testing.")
    parser.add_argument("--usfolds", default=[9, 10], type=int, nargs='+', help="If using US8K as the dataset," \
                    "you can determine the folds to use for testing.")

    return parser.parse_args()


def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
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
        
        # Print control via parser argument
        if args.print_out == True:
            print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # If it's a binary task, we compute auc
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

    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    _ , train_projs, train_lbls, _ , test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    if args.print_out == True:
        print(concept_bank.vectors, "concept bank vectors")
        print(train_projs, "train projs")
        print(weights, "weights")
        print(train_projs, "proj")
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    model_id = f"{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam_{args.lam}__alpha_{args.alpha}__seed_{args.seed}"
    model_id = f"{model_id}_target_{kwargs['target']}" if (args.dataset == "coco_stuff") else model_id
    model_path = os.path.join(args.out_dir, f"pcbm_{model_id}.ckpt")
    model_path = sub(":", "", model_path)
    torch.save(posthoc_layer, model_path)

    run_info_file = os.path.join(args.out_dir, f"rinf_pcbm_{model_id}.pkl")
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    if (num_classes > 1) and (args.print_out == True):
        # Prints the Top-5 Concept Weigths for each class if desired.
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}\n")
    print(run_info)


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

    # Execute main code
    train_runs(args, main, concept_bank, backbone, preprocess, mode="r")
    