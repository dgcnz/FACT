
# This script has some functions that can be used to verify the performances of the original models
import sys
sys.path.append("./models")

import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from models import clip_pl
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from data import get_dataset
from models import get_model
from training_tools import export
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Folder containing model/checkpoints.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seeds", default=[42, 43, 44], nargs='+', type=int, help="Random seeds")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--datasets", default=['cifar10'], nargs='+', type=str)
    parser.add_argument("--eval_all", action="store_true", default=False)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=None, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=2e-3, type=float, help="learning rate")
    parser.add_argument("--max-epochs", default=10, type=int, help="Maximum number of epochs.")
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


def eval_audio(args, seed):
    # setting the seed
    pl.seed_everything(seed, workers=True)
    
    accuracies = []
    _ , preprocess = get_model(args, backbone_name="audio")
    for fold in args.folds:
    
        new_args = deepcopy(args)
        new_args.escfold, new_args.usfolds = fold, fold

        train_loader, test_loader, _ , classes = get_dataset(new_args, preprocess)
        num_classes = len(classes)

        # define a checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                monitor='train_loss',
                                dirpath=args.out_dir + '/checkpoints/',
                                save_top_k=1,
                                mode='min',
                            )

        # first apply linear probing and instantiate the classifier module
        if args.checkpoint is not None:
            finetuner = clip_pl.AudioCLIPClassifierTrainer.load_from_checkpoint(args.checkpoint)
        else:
            finetuner = clip_pl.AudioCLIPClassifierTrainer(pretrained=True, n_classes=num_classes, lr=args.lr)

        trainer   = pl.Trainer(max_epochs=args.max_epochs, deterministic=True, 
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=1)
    
        trainer.fit(finetuner, train_loader)

        # load the best checkpoint
        best_model = clip_pl.AudioCLIPClassifierTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)

        # then evaluate the model
        results = trainer.test(model = best_model, dataloaders=test_loader)
        print('Current Accuracy: ' + str(results))
        accuracies.append(results[0]['test_accuracy'])

    return results # average accuracy over the batches
  

def get_features(model, loader, backbone:str="clip"):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            if backbone == "clip":
                features = model.encode_image(inputs.to(args.device))

            elif backbone == "audio":
                ((features, _, _), _), _ = model(audio=inputs.to(args.device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def eval_clip(args, seed, model, train_loader, test_loader):
    # Calculate the image features
    train_features, train_labels = get_features(model, train_loader)
    test_features, test_labels = get_features(model, test_loader)

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
                    classifier = LogisticRegression(random_state=args.seed, C=1/l2_lambda_list[l2_lambda_idx], max_iter=100, verbose=1, tol=1e-8)
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
    classifier = LogisticRegression(random_state=args.seed, C=C, max_iter=1000, verbose=1, tol=1e-8)
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


# def eval_audio(args, seed):

#     # AudioCLIP does not need the seed: The results should stay constant per dataset
#     # However, we do want to do cross-validation
#     accuracies = []
#     model, preprocess = get_model(args, backbone_name="audio")
#     print(model)
#     for fold in args.folds:

#         new_args = deepcopy(args)
#         new_args.escfold, new_args.usfolds = fold, fold
#         train_loader, test_loader, _ , _ = get_dataset(new_args, preprocess)
#         train_features, train_labels = get_features(model, train_loader, backbone="audio")
#         test_features, test_labels = get_features(model, test_loader, backbone="audio")

#         classifier = LogisticRegression(random_state=new_args.seed, C=0.001, max_iter=10000, verbose=1, tol=1e-9)
#         classifier.fit(train_features, train_labels)
#         predictions = classifier.predict(test_features)
#         print(classifier.coef_)
#         accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
#         print(f"Accuracy for fold {fold}: {accuracy:.3f}")
        
#         accuracies.append(accuracy)

#     final_acc = np.mean(accuracies)
#     print(f"Average Performance Across Folds: {final_acc:.3f}")

#     return final_acc # average accuracy over the batches


def eval_cifar(args, seed):
    model, preprocess = get_model(args, backbone_name="clip:RN50")
    train_loader, test_loader, _ , _ = get_dataset(args, preprocess)

    accuracy = eval_clip(args, seed, model, train_loader, test_loader)

    return accuracy


def eval_coco(args, seed):
    model, preprocess = get_model(args, backbone_name="clip:RN50")

    APs = []

    object_dict = {
          'car': 3,
          'bus': 6,
          'handbag': 31,
          'skis': 35,
          'snowboard': 36,
          'sports ball': 37,
          'baseball glove': 40,
          'skateboard': 41,
          'tennis racket': 43,
          'wine glass': 46,
          'cup': 47,
          'spoon': 50,
          'apple': 53,
          'potted plant': 64,
          'remote': 75,
          'keyboard': 76,
          'microwave': 78,
          'toaster': 80,
          'clock': 85,
          'hair drier': 89
    }

    #loop over the 20 targets
    for i in object_dict.values(): 
        train_loader, test_loader, _ , _ = get_dataset(args, preprocess, **{'target': i})
        AP = eval_clip(args, seed, model, train_loader, test_loader)
        APs.append(AP)
    
    mean_average_precision = np.mean(APs)
    print(f"Mean Average Precision = {mean_average_precision:.3f}")

    return mean_average_precision

def eval_cifar(args, seed):
    model, preprocess = get_model(args, backbone_name="clip:RN50")

    train_loader, test_loader, _ , classes = get_dataset(args, preprocess)
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
                accuracy_list = []
                for l2_lambda_idx in l2_lambda_idx_list:
                    classifier = LogisticRegression(random_state=args.seed, C=1/l2_lambda_list[l2_lambda_idx], max_iter=100, verbose=1)
                    classifier.fit(train_features_sweep, train_labels_sweep)
                    predictions = classifier.predict(val_features_sweep)
                    accuracy = np.mean((val_labels_sweep == predictions).astype(float)) * 100.
                    accuracy_list.append(accuracy)
                peak_idx = np.argmax(accuracy_list)
                peak_idx = l2_lambda_idx_list[peak_idx]
                return peak_idx

            l2_lambda_init_idx = [i for i,val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]
            peak_idx = find_peak(l2_lambda_init_idx)
            step_span = 2
            while step_span > 0:
                print(step_span, 'next iteration of the step span')
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
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")


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

    if "esc50" in args.datasets or args.eval_all:
        
        print("Evaluating for ESC-50")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "esc50"
        new_args.folds = [1, 2, 3, 4, 5]
        metrics = eval_per_seed(metrics, new_args, eval_audio, args.seeds)
    
    if "us8k" in args.datasets or args.eval_all:
        
        print("Evaluating for UrbanSound8K")
        print("========================")
        new_args = deepcopy(args)
        new_args.dataset = "us8k"
        new_args.folds = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        metrics = eval_per_seed(metrics, new_args, eval_audio, args.seeds)

    print("=======================\n")
    
    assert (metrics != {}), "It appears that none of the datasets you've specified are supported."

    # export results
    out_name = "evaluate_og_model.json"
    export.eval_export(out_name, metrics)
