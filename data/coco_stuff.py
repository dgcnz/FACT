from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from .constants import COCO_STUFF_DIR
import argparse
import json


# By default focuses on the 20 most biased classes mentioned in Singh et al., 2020
class COCODataset(Dataset):

    def __init__(self, datalist, image_dir, annot_dir, transform=None):
        """
        Arguments:
        datalist: a list object instance returned by 'load_coco_data' function below
        image_dir: directory of the folder containing all COCO-Stuff image folders
        annot_dir: directory of the corresponding annotation file
        transform: whether to apply any special transformation. Default = None
        """
        self.data = datalist
        self.transform = transform
        self.annot_dir = annot_dir
        self.image_dir = image_dir
        self.num_classes = 2 # By default these datasets are always for binary classification

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data[0]
        # # Trim unnecessary paths

        # idx = img_path.split('/').index('CUB_200_2011')
        # img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
        # img = Image.open(img_path).convert('RGB')

        # class_label = img_data['class_label']
        # if self.transform:
        #     img = self.transform(img)
        img = []
        class_label = []

        return img, class_label

def cid_to_class(pathtxt, target_labs): # "cid = COCO ID"
    labeldict = {}
    with open(pathtxt) as f:
        for line in f.readlines():
            idx, label = line.strip("\n").split(": ")
            if (label in target_labs) or (target_labs is None):
                labeldict[idx] = label

    return labeldict

def load_coco_data(image_dir, annot_dir, transform=None):

    f = open(annot_dir)
    data = json.load(f)

    return NotImplemented

# For testing:
# def config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
#     parser.add_argument("--dataset-name", default="cub", type=str)
#     parser.add_argument("--out-dir", required=True, type=str)
#     parser.add_argument("--device", default="cuda", type=str)
#     parser.add_argument("--seed", default=1, type=int, help="Random seed")
#     parser.add_argument("--num-workers", default=4, type=int, help="Number of workers in the data loader.")
#     parser.add_argument("--batch-size", default=100, type=int, help="Batch size in the concept loader.")
#     parser.add_argument("--C", nargs="+", default=[0.01, 0.1], type=float, help="Regularization parameter for SVMs.")
#     parser.add_argument("--n-samples", default=50, type=int, 
#                         help="Number of positive/negative samples used to learn concepts.")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = config()
#     coco_data = load_coco_data(args)

# TO BE IMPLEMENTED
