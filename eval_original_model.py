
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score




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
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--C", default=None, type=float, help="Inverse of regularization strength. Smaller values cause stronger regularization.")
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


def eval_cifar_old(args, seed):
    # setting the seed
    pl.seed_everything(seed, workers=True)

    _ , preprocess = get_model(args, backbone_name="clip:RN50")

    train_loader, test_loader, _ , classes = get_dataset(args, preprocess)
    num_classes = len(classes)

    #create a validation set from the training set
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])

    #create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.num_workers)

    print(f"Evaluating for seed: {seed}")

    #define a checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                            monitor='val_loss',
                            dirpath=args.out_dir + '/checkpoints/',
                            save_top_k=1,
                            mode='min',
                        )

    # first apply linear probing and instantiate the classifier module
    if args.checkpoint is not None:
        finetuner = clip_pl.CLIPClassifierTrainer.load_from_checkpoint(args.checkpoint)
    else:
        finetuner = clip_pl.CLIPClassifierTrainer("RN50", n_classes=num_classes, lr=args.lr)

    trainer   = pl.Trainer(max_epochs=args.max_epochs, deterministic=True, 
                    callbacks=[checkpoint_callback],
                    check_val_every_n_epoch=1) #EarlyStopping(monitor="val_loss", mode="min"), 
    
    # I have to actually pass a validation loader to this for it to work correctly. otherwise it will just use the train steps always. 
    trainer.fit(finetuner, train_loader, val_loader)

    # load the best checkpoint
    best_model = clip_pl.CLIPClassifierTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)

    # then evaluate the model
    results = trainer.test(model = best_model, dataloaders=test_loader)
    print('Current Accuracy: ' + str(results))

    return results # average accuracy over the batches

def eval_clip(args, model, train_loader, test_loader, classes):
    num_classes = len(classes)

    def get_features(loader):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader):
                features = model.encode_image(images.to(args.device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train_loader)
    test_features, test_labels = get_features(test_loader)

    #split train set into train and validation in numpy arrays
    train_size = int(0.8 * len(train_features))
    train_features_sweep, val_features_sweep = np.split(train_features, [train_size])
    train_labels_sweep, val_labels_sweep = np.split(train_labels, [train_size])

    # do a hyperparameter sweep to find the best regularization strength lambda.
    if args.C is None:     
        def hyperparameter_sweep():
            print("Performing hyperparameter sweep to find the best regularization strength lambda.")
            l2_lambda_list = np.logspace(-6, 6, num=97).tolist()
            
            def find_peak(l2_lambda_idx_list):
                """Calculate accuracy on all indexes and return the peak index"""
                metric_list = []
                for l2_lambda_idx in l2_lambda_idx_list:
                    classifier = LogisticRegression(random_state=args.seed, C=1/l2_lambda_list[l2_lambda_idx], max_iter=100, verbose=1)
                    classifier.fit(train_features_sweep, train_labels_sweep)
                    predictions = classifier.predict(val_features_sweep)
                    if args.dataset == "coco_stuff":
                        metric = average_precision_score(val_labels_sweep, predictions) 
                    else:
                        metric = np.mean((val_labels_sweep == predictions).astype(float)) * 100.
                    metric_list.append(metric)
                peak_idx = np.argmax(metric_list)
                peak_idx = l2_lambda_idx_list[peak_idx]
                return peak_idx

            l2_lambda_init_idx = [i for i,val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]
            peak_idx = find_peak(l2_lambda_init_idx)
            step_span = 2
            while step_span > 0:
                print(step_span, 'wtffffff why does this run for infinity time')
                left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list)-1)
                peak_idx = find_peak([left, peak_idx, right])
                step_span //= 2
                print('current best lambda', l2_lambda_list[peak_idx])
            return l2_lambda_list[peak_idx]

        lambda_best = hyperparameter_sweep()
        C = 1 / lambda_best
        print(C, 'best C')

    else:
        C = args.C

    # Perform logistic regression
    classifier = LogisticRegression(random_state=args.seed, C=C, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    if args.dataset == "coco_stuff":
        metric = average_precision_score(test_labels, predictions)
        print(f"Average precision = {metric:.3f}")
    else:
        metric = np.mean((test_labels == predictions).astype(float)) * 100.
        print(f"Accuracy = {metric:.3f}")
    
    return metric

def eval_cifar(args, seed):
    model, preprocess = get_model(args, backbone_name="clip:RN50")
    train_loader, test_loader, _ , classes = get_dataset(args, preprocess)

    accuracy = eval_clip(args, seed, model, train_loader, test_loader, classes)

    return accuracy

def eval_coco(args, seed):
    model, preprocess = get_model(args, backbone_name="clip:RN50")

    APs = []

    #loop over the 20 targets
    for i in range(20): 
        train_loader, test_loader, _ , classes = get_dataset(args, preprocess, **{'target': i})
        AP = eval_clip(args, seed, model, train_loader, test_loader, classes)
        APs.append(AP)
    
    mean_average_precision = np.mean(APs)
    print(f"Mean Average Precision = {mean_average_precision:.3f}")

    return mean_average_precision

    




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
        
        print("Evaluating for COCO-stuff")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "coco_stuff"
        metrics = eval_per_seed(metrics, new_args, eval_coco, args.seeds)
    
    print("=======================\n")
    
    assert (metrics != {}), "It appears that none of the datasets you've specified are supported."

    # export results
    out_name = "evaluate_og_model.json"
    export.eval_export(out_name, metrics)
