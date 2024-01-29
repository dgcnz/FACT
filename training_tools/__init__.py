import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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


def load_coco_data(annot_dir, target:int, n_samples:int=500,
                   transform=None, batch_size:int=1, seed:int=42):
    """
    Arguments:
    annot_dir: directory of the corresponding annotation file
    transform: whether to apply any special transformation. Default = None
    target: the class index of the target class (see below for guide)
    n_samples: number of samples to take (default 250, should be 500 for training)
    seed: random seed for sampling

    Outputs:
    dataloader: torch dataloader instance containing the data
    """
    # set the seeds
    np.random.seed(seed)

    # open the file
    f = open(annot_dir)
    data = json.load(f)

    # even split between training and test data
    npc = n_samples // 2
        
    # select images based on whether not the image has a segmentation of the target object
    pos_data = []
    neg_data = []
    for imgc in data.keys():
        if target in data[imgc]:
            pos_data.append(imgc)
        else:
            neg_data.append(imgc)

    # sample with replacement if too little data is present    
    replace = True if (len(pos_data) < npc) else False
    pos_sample = np.random.choice(pos_data, size=npc, replace=replace)
    neg_sample = np.random.choice(pos_data, size=npc, replace=replace)
    
    pos_sample = [[pos_sample[idx], 1] for idx in range(npc)]
    neg_sample = [[neg_sample[idx], 0] for idx in range(npc)]

    datalist = pos_sample + neg_sample
    dataset = COCODataset(datalist, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


# For testing
if __name__ == "__main__":
    ltr_path = os.path.join(COCO_STUFF_DIR, "labels_train.json")
    lt_path = os.path.join(COCO_STUFF_DIR, "labels_val.json")

    train_coco = load_coco_data(ltr_path, target=41, n_samples=500)
    test_coco = load_coco_data(lt_path, target=41, n_samples=250)

    first_x, first_y = next(iter(test_coco))
    
    print(first_x)
    print(first_y)