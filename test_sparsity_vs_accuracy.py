import os
import pickle
import numpy as np
import torch
import argparse
from re import sub
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Folder containing model/checkpoints.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--strengths", default=None, type=float, nargs='+', help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--targets", default=[3, 6, 31, 35, 36, 37, 40, 41, \
                                             43, 46, 47, 50, 53, 64, 75, 76, 78, 80, 85, 89], \
                                             type=int, nargs='+', help="target indexes for cocostuff")

    return parser.parse_args()


def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=2) # TODO: change to OLS package function such that I can do tests and stuff on it. essentially a logistic regression. 
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
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    
    return run_info, classifier.coef_, classifier.intercept_


def main(args, concept_bank, backbone, preprocess):
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    
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
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    model_id = f"{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}"
    model_path = os.path.join(args.out_dir, f"pcbm_{model_id}.ckpt")
    model_path = sub(":", "", model_path)
    torch.save(posthoc_layer, model_path)

    run_info_file = os.path.join(args.out_dir, f"run_info-pcbm_{model_id}.pkl")
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    if num_classes > 1:
        # Prints the Top-5 Concept Weigths for each class.
        print(posthoc_layer.analyze_classifier(k=5))

    run_info["sparsity"] = posthoc_layer.get_sparsity()
    run_info["sum"] = np.abs(weights.sum())

    print(f"Model saved to : {model_path}")
    print(run_info)

    return run_info


def plot_sparsity(args, metrics, sparsities, metric_name):
    import matplotlib.pyplot as plt
    print("metric used:", metric_name)
    print(metrics)
    plt.plot(args.strengths, sparsities)
    plt.grid()
    plt.ylabel("N non-zero weights")
    plt.ylabel('Regularization strength')
    plt.savefig(f"{args.out_dir}/sparsity.png")
    plt.show()

    print(f'Figure save to : {args.out_dir}/sparsity.png')


def plot_sum(args, metrics, sums, metric_name):
    import matplotlib.pyplot as plt
    print("metric used:", metric_name)
    print(metrics)
    plt.plot(args.strengths, sums)
    plt.grid()
    plt.ylabel("Sum of weights")
    plt.xlabel('Regularization strength')
    plt.savefig(f"{args.out_dir}/sum.png")
    plt.show()

    print(f'Figure save to : {args.out_dir}/sum.png')
    

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
    
    classes = get_dataset(args, preprocess)[3]
    print(classes)
    metrics = []
    sums= []
    metric_name = "Accuracy"

    sparsities = []

    print(f'number of concepts: {len(all_concept_names)}')
    print(f'number of classes: {len(classes)}')

    for strength in args.strengths:

        args.lam = strength/(len(all_concept_names)*len(classes))

        run_info = main(args, concept_bank, backbone, preprocess)

        if "test_auc" in run_info:
            print("AUC used")
            metric_name = "AUC"
            metric = run_info['test_auc']

        else:
            print("Accuracy used")
            metric_name = "Accuracy"
            metric = run_info['test_acc']

        metrics.append(metric)
        sums.append(run_info['sum'])
        sparsities.append(run_info['sparsity'])

    plot_sparsity(args, metrics, sparsities, metric_name)
    plot_sum(args, metrics, sums, metric_name)

