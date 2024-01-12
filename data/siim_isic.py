from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from .constants import COCO_STUFF_DIR
import argparse

# Currently based on CUB Dataset class (am researching what components are needed)
class ISICDataset(Dataset):

    def __init__(self, image_dir, meta_dir, select_rand, num_classes, transform=None):
        """
        Arguments:
        annot_dir: directory of the corresponding annotation file
        select_rand: because each image has 3 distinct captions, determines how the labels are selected
        image_dir: directory of the folder containing all COCO-Stuff image folders
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        f = open(meta_dir)
        self.data = pd.read_csv(f) # CSV object containing all metadata of set
        f.close()

        self.transform = transform
        self.select_rand = select_rand
        self.image_dir = image_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_data = self.data[idx]
        # img_path = img_data['img_path']
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
    return NotImplemented

def load_siim_data(image_dir, meta_dir, transform=None):
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