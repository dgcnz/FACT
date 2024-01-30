import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from data import get_dataset
from concepts import ConceptBank
from models import PCBMUserStudy, get_model
from training_tools import load_or_compute_projections
import copy
import time
import itertools


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face Token", required=True)
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="task_1_bed_dog", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--print-out", default=True, type=bool)
    parser.add_argument("--greedy-pruning", default=False, type=bool)
    parser.add_argument("--prune", default="dog", type=str)   

    args.pruning = [concept for concept in args.prune.split(',')]
    args.seeds = [int(seed) for seed in args.seeds.split(',')]
    return parser.parse_args()

def run_linear_probe(args, train_data, test_data, classes):
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
        cls = classes[lbl]
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        
        # Print control via parser argument
        if args.print_out == True:
            print(f"{lbl, cls}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }
    
    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
                                              
    return run_info, classifier.coef_, classifier.intercept_

def evaluate_model_accuracy(model, test_data, args):
    test_features, test_labels = test_data
    predictions = model.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    return accuracy

def get_concept_index(concept_name, concept_bank):
    return concept_bank.concept_names.index(concept_name)

def get_class_index(class_name, idx_to_class):
    for idx, name in idx_to_class.items():
        if name == class_name:
            return idx
    return None

def main(args, concept_bank, backbone, preprocess):
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    torch.manual_seed(args.seed)
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)
    
    # Initialize the PCBM module.
    posthoc_layer = PCBMUserStudy(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls), classes)
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    model_id = f"{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}"
    model_path = os.path.join(args.out_dir, f"pcbm_{model_id}.ckpt")
    torch.save(posthoc_layer, model_path)

    run_info_file = os.path.join(args.out_dir, f"run_info-pcbm_{model_id}.pkl")
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    if (num_classes > 1):
        # Prints the Top-10 Concept Weigths for each class if desired.
        print(posthoc_layer.analyze_classifier(k=10))
        import pandas as pd
        _, top_concepts = posthoc_layer.analyze_classifier_withResults(k=10)
        df = pd.DataFrame(top_concepts)
        csv_path = os.path.join(args.out_dir, args.dataset+"_weights.csv")
        df.to_csv(csv_path, index=False)
        print(f"Analysis saved to {csv_path}")

    print(f"Model saved to : {model_path}")
    print(run_info)
    

    if args.greedy_pruning:
        best_accuracy = evaluate_model_accuracy(posthoc_layer, (test_projs, test_lbls))
        best_state = copy.deepcopy(posthoc_layer.state_dict())
        best_combination = []

        start_time = time.time() 
        # Greedy search for best pruning
        for r in range(1, len(top_concepts) + 1):
            for combination in itertools.combinations(top_concepts, r):
                for concept in combination:
                    posthoc_layer.prune(get_concept_index(concept), get_class_index(concept))

                current_accuracy = evaluate_model_accuracy(posthoc_layer, (test_projs, test_lbls))

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_state = copy.deepcopy(posthoc_layer.state_dict())
                    best_combination = combination
                else:
                    #Revert to previous best state
                    posthoc_layer.load_state_dict(best_state)

        end_time = time.time() 
        pruning_time = end_time - start_time
        print(f"Greeding pruning process took {pruning_time:.2f} seconds.")
        print(f"Best pruning combination: {best_combination}")
        print(f"Best accuracy after pruning: {best_accuracy}")

    if args.pruning:
        for concept_to_prune in args.pruning:
            posthoc_layer.prune(get_concept_index(concept_to_prune), get_class_index(concept_to_prune))
            get_concept_index(concept_to_prune)
            get_class_index(concept_to_prune)
        run_info_after_pruning, pruning_weights, _ = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls), classes)
        print("Performance before pruning:", run_info)
        print("Performance after pruning:", run_info_after_pruning)

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
    # Execute main code
    #main(args, concept_bank, backbone, preprocess)
    for seed in args.seeds:
        print(f"Seed: {seed}")
        args.seed = seed
        run_info = main(args, concept_bank, backbone, preprocess)

        if "test_auc" in run_info:
            print("auc used")
            metric = run_info['test_auc']

        else:
            print("acc used")
            metric = run_info['test_acc']

        metric_list.append(metric)
