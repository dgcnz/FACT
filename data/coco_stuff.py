import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .constants import COCO_STUFF_DIR


# By default focuses on the 20 most biased classes mentioned in Singh et al., 2020
class COCODataset(Dataset):

    def __init__(self, datalist, transform=None):
        """
        Arguments:
        datalist: a list object instance returned by 'load_coco_data' function below
        transform: whether to apply any special transformation. Default = None
        """
        self.data = datalist
        self.transform = transform
        self.num_classes = 2 # By default these datasets are always for binary classification

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = os.path.join(COCO_STUFF_DIR, img_data[0])
        img = Image.open(img_path).convert('RGB')
        class_label = img_data[1]
        if self.transform:
            img = self.transform(img)

        return img, class_label

# function for converting 'coco_target_indexes.txt" to a dict for binary classification
def cid_to_class(pathtxt, target_labs, target_idx): # "cid = COCO ID"
    labeldict = {}
    with open(pathtxt) as f:
        for line in f.readlines():
            label, idx = line.strip("\n").split(": ")
            if (label in target_labs) or (target_labs is None):
                if int(idx) == target_idx:
                   labeldict[1] = label
                else:
                   labeldict[0] = 'other'

    return labeldict


def load_coco_data(train_dir, test_dir, target:int, n_train:int=500, n_test:int=250,
                   transform=None, batch_size:int=1, seed:int=42):
    """
    Arguments:
    train_dir: directory of the corresponding training annotation file
    test_dir: directory of the corresponding test annotation file
    transform: whether to apply any special transformation. Default = None
    target: the class index of the target class (see below for guide)
    n_train: number of samples to take for training (default 500)
    n_test: number of samples to take for training (default 250)
    seed: random seed for sampling

    Outputs:
    dataloader: torch dataloader instance containing the data
    """
    # set the seeds
    np.random.seed(seed)

    # open the files
    f_tr = open(train_dir)
    data_tr = json.load(f_tr)

    f_ts = open(test_dir)
    data_ts = json.load(f_ts)

    # even split between training and test data
    assert (n_train % 2 == 0), "Amount of training samples is odd. Please use an even number!"
    assert (n_test % 2 == 0), "Amount of test samples is odd. Please use an even number!"
    n_htr = n_train // 2
    n_hts = n_test // 2
        
    # select images based on whether not the image has a segmentation of the target object
    pos_train_data = []
    neg_train_data = []
    pos_test_data = []
    neg_test_data = []
    for imgc in data_tr.keys():
        if target in data_tr[imgc]:
            pos_train_data.append(imgc)
        else:
            neg_train_data.append(imgc)

    for imgc in data_ts.keys():
        if target in data_ts[imgc]:
            pos_test_data.append(imgc)
        else:
            neg_test_data.append(imgc)

    # sample with replacement if too little data is present
    replace_pos = True if (len(pos_train_data) < n_htr) else False
    replace_neg = True if (len(neg_train_data) < n_htr) else False
    pos_sample_tr = np.random.choice(pos_train_data, size=n_htr, replace=replace_pos)
    neg_sample_tr = np.random.choice(neg_train_data, size=n_htr, replace=replace_neg)

    replace_pos = True if (len(pos_test_data) < n_hts) else False
    replace_neg = True if (len(neg_test_data) < n_hts) else False
    pos_sample_ts = np.random.choice(pos_test_data, size=n_hts, replace=replace_pos)
    neg_sample_ts = np.random.choice(neg_test_data, size=n_hts, replace=replace_neg)

    train_data = [[pos_sample_tr[idx], 1] for idx in range(n_htr)] + [[neg_sample_tr[idx], 0] for idx in range(n_htr)]
    test_data  = [[pos_sample_ts[idx], 1] for idx in range(n_hts)] + [[neg_sample_ts[idx], 0] for idx in range(n_hts)]

    train = COCODataset(train_data, transform)
    test = COCODataset(test_data, transform)
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader


# For testing
if __name__ == "__main__":
    ltr_path = os.path.join(COCO_STUFF_DIR, "labels_train.json")
    lts_path = os.path.join(COCO_STUFF_DIR, "labels_val.json")

    train_coco, test_coco = load_coco_data(ltr_path, lts_path, target=3)
    train_coco, test_coco = load_coco_data(ltr_path, lts_path, target=6)
    train_coco, test_coco = load_coco_data(ltr_path, lts_path, target=31)
    train_coco, test_coco = load_coco_data(ltr_path, lts_path, target=64)
    train_coco, test_coco = load_coco_data(ltr_path, lts_path, target=41)

    # first_x, first_y = next(iter(test_coco))
    
    # print(first_x)
    # print(first_y)
