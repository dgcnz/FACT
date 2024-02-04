import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from data import get_dataset
from concepts import ConceptBank
from models import get_model
from models.pcbm_utils_prune import PCBMUserStudy
from training_tools import load_or_compute_projections, export
import copy
import time
import itertools
import pandas as pd
import json

greedy_pruning_results = []
user_pruning_results = []
random_pruning_results = []

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
    parser.add_argument("--random-pruning", default=False,type=bool)
    parser.add_argument("--pruning-class",  default="", type=str)
    parser.add_argument("--prune", default="", type=str)   
    parser.add_argument("--number-of-concepts-to-prune", default="",type=str)

    args = parser.parse_args()
    if args.prune:
      args.prune = json.loads(args.prune)
    args.seeds = [int(seed) for seed in args.seeds.split(',')]
    if args.number_of_concepts_to_prune:
      args.number_of_concepts_to_prune = [int(numbr) for numbr in args.number_of_concepts_to_prune.split(',')]
    return args

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

    if (args.print_out):
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
    index_of_class = get_class_index(args.pruning_class, idx_to_class)
    class_original_acc = run_info['cls_acc']['test'][index_of_class] *100.
    model_accuracy = run_info['test_acc']  
    current_model_state = copy.deepcopy(posthoc_layer.state_dict())
   
    best_greedy_model = PCBMUserStudy(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    best_greedy_model = best_greedy_model.to(args.device)
    if args.greedy_pruning:
        concepts = []
        for tc in top_concepts:
            if tc['class'] == args.pruning_class:
                concepts.append(tc)

        best_pruning_acc = 0
        start_time = time.time() 
        # Greedy search for best pruning
        
        for r in args.number_of_concepts_to_prune:
            best_pruning_acc = 0
            best_combination = None
            pruned_class_acc = 0
            for combination in itertools.combinations(concepts, r):
                for concept in combination:
                    posthoc_layer.prune(get_concept_index(concept['concept'], concept_bank), index_of_class)

                total_accuracy, class_acc = posthoc_layer.test_step((test_projs, test_lbls), args.device)
                if total_accuracy> best_pruning_acc:
                    best_pruning_acc = total_accuracy
                    best_combination = combination
                    pruned_class_acc = class_acc[index_of_class].item()
                    best_greedy_model.load_state_dict(posthoc_layer.state_dict())

                posthoc_layer.load_state_dict(current_model_state)
            greedy_pruning_results.append({
                'seed': args.seed,
                'concepts pruned':r,
                'combination': best_combination,
                'original acc':model_accuracy,
                'accuracy': best_pruning_acc,
                'delta':  best_pruning_acc - model_accuracy,
                'class original acc':class_original_acc,
                'class acc':pruned_class_acc,
                'class delta':pruned_class_acc-class_original_acc
            })
            model_id = f"concepts_pruned:{r}__{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}"
            model_path = os.path.join(args.out_dir, f"greedy_pcbm_{model_id}.ckpt")
            torch.save(best_greedy_model, model_path)
            run_info_file = os.path.join(args.out_dir, f"run_info-greedy_pcbm_{model_id}.pkl")

        end_time = time.time() 
        pruning_time = end_time - start_time
        print(f"Greeding pruning process took {pruning_time:.2f} seconds.")

    if args.random_pruning:
        import random
        concepts = []
        for tc in top_concepts:
            if tc['class'] == args.pruning_class:
                concepts.append(tc)

        best_pruning_acc = 0
        start_time = time.time() 
        # Greedy search for best pruning
        pruning_to_select =[]
        for r in args.number_of_concepts_to_prune:
            best_pruning_acc = 0
            best_combination = None
            pruned_class_acc = 0
            for combination in itertools.combinations(concepts, r):
                for concept in combination:
                    posthoc_layer.prune(get_concept_index(concept['concept'], concept_bank), index_of_class)

                total_accuracy, class_acc = posthoc_layer.test_step((test_projs, test_lbls), args.device)
                pruned_class_acc = class_acc[index_of_class].item()
                pruning_to_select.append({
                'seed': args.seed,
                'concepts pruned':r,
                'combination': combination,
                'original acc':model_accuracy,
                'accuracy': total_accuracy,
                'delta':  total_accuracy - model_accuracy,
                'class original acc':class_original_acc,
                'class acc':pruned_class_acc,
                'class delta':pruned_class_acc-class_original_acc
                })
                model_id = f"concepts_pruned:{r}__{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}"
                model_path = os.path.join(args.out_dir, f"random_pcbm_{model_id}.ckpt")
                torch.save(posthoc_layer, model_path)
                run_info_file = os.path.join(args.out_dir, f"run_info-random_pcbm_{model_id}.pkl")
                posthoc_layer.load_state_dict(current_model_state)
            random_pruning_results.append(random.choice(pruning_to_select))

        end_time = time.time() 
        pruning_time = end_time - start_time
        print(f"Random pruning process took {pruning_time:.2f} seconds.")
    
    if args.prune:
        for answer in args.prune:
            concepts = [concept for concept in answer.split(',')]
            for concept_to_prune in concepts:
                posthoc_layer.prune(get_concept_index(concept_to_prune, concept_bank), get_class_index(args.pruning_class, idx_to_class))

            pruning_accuracy, class_acc = posthoc_layer.test_step((test_projs, test_lbls), args.device)
            pruned_class_acc = class_acc[index_of_class].item()

            print("Performance before pruning:", model_accuracy)
            print("Performance after pruning:", pruning_accuracy)
            print("Class performance after pruning:", pruned_class_acc)
            model_id = f"concepts_pruned:{len(concepts)}__{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}"
            model_path = os.path.join(args.out_dir, f"user_pcbm_{model_id}.ckpt")
            torch.save(posthoc_layer, model_path)
            run_info_file = os.path.join(args.out_dir, f"run_info-user_pcbm_{model_id}.pkl")
            posthoc_layer.load_state_dict(current_model_state)
            posthoc_layer.load_state_dict(current_model_state)

            user_pruning_results.append({
                    'seed': args.seed,
                    'number concepts': len(concepts),
                    'original acc':model_accuracy,
                    'accuracy': pruning_accuracy,
                    'delta':  pruning_accuracy - model_accuracy,
                    'class original acc':class_original_acc,
                    'class acc':pruned_class_acc,
                    'class delta':pruned_class_acc-class_original_acc,
                    'accuracy improved': (pruning_accuracy - model_accuracy) >0
            })
    
    return run_info, class_original_acc


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
    class_list = []
    # Execute main code
    #main(args, concept_bank, backbone, preprocess)
    for seed in args.seeds:
        print(f"Seed: {seed}")
        args.seed = seed
        run_info, cls_acc = main(args, concept_bank, backbone, preprocess)

        if "test_auc" in run_info:
            metric = run_info['test_auc']

        else:
            metric = run_info['test_acc']

        metric_list.append(metric)
        class_list.append(cls_acc)

    if greedy_pruning_results:
        results_df = pd.DataFrame(greedy_pruning_results)
        csv_path = os.path.join(args.out_dir, f"greedy_pruning_results_{args.dataset}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Global greedy pruning results saved to {csv_path}")
    if random_pruning_results:
        results_df = pd.DataFrame(random_pruning_results)
        csv_path = os.path.join(args.out_dir, f"random_pruning_results_{args.dataset}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Random pruning results saved to {csv_path}")
    if user_pruning_results:
        results_df = pd.DataFrame(user_pruning_results)
        csv_path = os.path.join(args.out_dir, f"users_pruning_results_{args.dataset}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"User pruning results saved to {csv_path}")

    out_name = "UserStudy_results_"+args.dataset
    export.export_to_json(out_name, metric_list)
    print('Spurious class metrics >>>')
    print(class_list)
    print('Mean : {}'.format(np.mean(class_list)))
    print('Std : {}'.format(np.std(class_list)))