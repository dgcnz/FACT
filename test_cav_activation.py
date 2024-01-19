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

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections, export
from torch.utils.data import DataLoader, random_split
from data import get_concept_loaders


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default='42', type=str, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=None, type=float, help="Regularization strength.")
    parser.add_argument("--n-samples", default=50, type=int, 
                        help="Number of positive/negative samples used to learn concepts.")

    ## arguments for the different projection matrix weights
    parser.add_argument("--random_proj", action="store_true", default=False, help="Whether to use random projection matrix")

    parser.add_argument("--identity_proj", action="store_true", default=False, help="Whether to use identity projection matrix")
    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(',')]
    return args

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

    # Get the true concepts for the dataset, per image
    concept_loaders = get_concept_loaders(args.dataset, preprocess, n_samples=args.n_samples, batch_size=args.batch_size, 
                                          num_workers=args.num_workers, seed=args.seed)
    
    positive_projection_magnitude_per_concept = {}
    negative_projection_magnitude_per_concept = {}
    
    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    for concept_name, loaders in concept_loaders.items():
        pos_loader, neg_loader = loaders['pos'], loaders['neg']
        train_embs_pos, train_projs_pos = load_or_compute_projections(args, backbone, posthoc_layer, pos_loader, test_loader, compute = True, self_supervised=True)

        train_embs_neg, train_projs_neg = load_or_compute_projections(args, backbone, posthoc_layer, neg_loader, test_loader, compute = True, self_supervised=True)

        # Select only the projection of our current concept of interest
        assert train_projs_pos.shape[1] == len(concept_bank.concept_names), "wrong dimension selected for concept of interest"
        train_projs_pos = train_projs_pos[:, concept_bank.concept_names.index(concept_name)] 
        train_projs_neg = train_projs_neg[:, concept_bank.concept_names.index(concept_name)]

        # Compute the average
        positive_projection_magnitude_per_concept[concept_name] = np.mean(train_projs_pos)
        negative_projection_magnitude_per_concept[concept_name] = np.mean(train_projs_neg)

    print(positive_projection_magnitude_per_concept)
    print(negative_projection_magnitude_per_concept)

    #We get the total average activation for pos and neg 
    total_average_pos_activation = np.mean(list(positive_projection_magnitude_per_concept.values()))
    total_average_neg_activation = np.mean(list(negative_projection_magnitude_per_concept.values()))

    print(f"total average pos activation: {total_average_pos_activation}")
    print(f"total average neg activation: {total_average_neg_activation}")

    run_info = {}
    run_info['total_average_pos_activation'] = total_average_pos_activation
    run_info['total_average_neg_activation'] = total_average_neg_activation

    return run_info

if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    #to be completely robust to oversight, set all attributes (/ concept names) of the concept bank class to None
    shape = concept_bank.vectors.shape

    #change the following three attributes of the ConceptBank class
    #self.cavs = concept_bank.vectors
    #self.intercepts = concept_bank.intercepts -> seem svm based thing, why use these when you use clip concepts?
    #self.norms = concept_bank.norms

    if args.random_proj:
        concept_bank.vectors = None
        concept_bank.intercepts = None
        concept_bank.norms = None
        concept_bank.margin_info = None
        print(concept_bank.vectors)

        concept_bank.vectors = torch.randn(shape).to(args.device)
        print(concept_bank.vectors)
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

    
    print(f'concept vectors matrix rank is {torch.linalg.matrix_rank(concept_bank.vectors)}')

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    pos = []
    neg = []
    og_out_dir = args.out_dir

    for seed in args.seeds:
        print(f"Seed: {seed}")
        args.seed = seed
        args.out_dir = og_out_dir 
        run_info = main(args, concept_bank, backbone, preprocess)

        pos.append(run_info['total_average_pos_activation'])
        neg.append(run_info['total_average_neg_activation'])