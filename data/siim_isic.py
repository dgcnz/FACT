from glob import glob
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from .constants import SIIM_DATA_DIR
import argparse

# Currently based on CUB Dataset class (am researching what components are needed)
class ISICDataset(Dataset):

    def __init__(self, datalist, transform=None):
        """
        Arguments:
        annot_dir: directory of the corresponding annotation file
        datalist: a list object instance returned by 'data_select' function above
        transform: whether to apply any special transformation. Default = None
        """
        self.data = datalist
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data[0]
        img = Image.open(img_path).convert('RGB')

        class_label = img_data[1]
        if self.transform:
            img = self.transform(img)

        return img, class_label

# Function for preparing the data from the data (2500 in total: 80% benign, 20% malignant)
def prepare_data(df, n_train:int=2000, n_test:int=500, ratio:float=0.2, seed:int=42):
    """
    Arguments:
    df: Pandas dataframe object
    n_train, n_test: the amount of training and test images to sample respectively
    ratio: percentage of data to use for testing
    seed: Random seed for train test split

    Outputs:
    train_data, test_data: Lists in the form of a PyTorch datalist; '[[img_path1, img_label1], [img_path2, img_label2], ...]'
    """
    # Obtaining a balanced number of samples from the dataset
    n_train_ben = int(n_train * (1 - ratio)) # 1600
    n_train_mal = int(n_train * ratio)       # 400
    n_test_ben = int(n_test * (1 - ratio))   # 400
    n_test_mal = int(n_test * ratio)         # 100

    # Taking n samples from the dataset (sampled in a way which prevents overlap)
    train_ben = df.loc[df['benign_malignant'] == 'benign'][:n_train_ben]
    train_mal = df.loc[df['benign_malignant'] == 'malignant'][:n_train_mal]
    test_ben = df.loc[df['benign_malignant'] == 'benign'][n_train_ben:n_train_ben+n_test_ben]
    test_mal = df.loc[df['benign_malignant'] == 'malignant'][n_train_mal:n_train_mal+n_test_mal]
    clean_df = pd.concat([train_ben, train_mal, test_ben, test_mal]) # Combining and converting the dataset

    datalist = [list(clean_df['image_name']), list(clean_df['benign_malignant'])] # PyTorch data list in the format of '[[img_paths], [img_labels]]'
    X_train, X_test, y_train, y_test = train_test_split(datalist[0], datalist[1], 
                                                        stratify=datalist[1], test_size=ratio, 
                                                        random_state=seed)
    
    train_data = [[X_train[idx], y_train[idx]] for idx in range(len(X_train))]
    test_data = [[X_test[idx], y_test[idx]] for idx in range(len(X_test))]

    return train_data, test_data

def load_siim_data(meta_dir, transform=None, seed:int=42):

    df = pd.read_csv(meta_dir)[['image_name', 'benign_malignant']]
    train_dir = os.path.join(SIIM_DATA_DIR, "train")
    df['image_name'] = train_dir + "/" + df['image_name'] + ".jpg"
    train_data, test_data = prepare_data(df)

    train_data = ISICDataset(train_data, transform)
    test_data = ISICDataset(test_data, transform)

    train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data)

    return train_loader, test_loader

# For testing:
if __name__ == "__main__":

    meta_path = os.path.join(SIIM_DATA_DIR, "isic_metadata.csv")
    train_siim, test_siim = load_siim_data(meta_path)
