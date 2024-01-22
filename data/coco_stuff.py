import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import COCO_STUFF_DIR


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
        img_path = img_data[0]
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)
        class_label = img_data[1]
        if self.transform:
            img = self.transform(img)

        return img, class_label

def cid_to_class(pathtxt, target_labs): # "cid = COCO ID"
    labeldict = {}
    with open(pathtxt) as f:
        for line in f.readlines():
            idx, label = line.strip("\n").split(": ")
            if (label in target_labs) or (target_labs is None):
                labeldict[idx] = label

    return labeldict

def idx2img(json_file, image_dir):
    """
    Arguments:
    json_file: the annotation JSON file loaded by load_coco_data below
    image_dir: the image dir passed to load_coco_data
    
    Outputs:
    img_dict: dictionary containing mappings of '{<image index>: image_dir + <image filename>}'
    """
    img_dict = {}
    for data in json_file['images']:

        id = data['id']
        name = data['file_name']
        img_dict[id] = os.path.join(image_dir, name)

    return img_dict

def load_coco_data(image_dir, annot_dir, target:int, n_samples:int=500,
                   transform=None, batch_size:int=1, seed:int=42):
    """
    Arguments:
    image_dir: directory containing the images to load (either train2017/val2017)
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

    # get the annotation index to filename mapping
    img_dict = idx2img(data, image_dir)

    # even split between training and test data
    npc = n_samples // 2
    img_idx_dict = {}

    # extract the data
    for segment in data['annotations']:
        class_id = segment['category_id']
        image_id = img_dict[segment['image_id']]
        if image_id not in img_idx_dict.keys():
            img_idx_dict[image_id] = [class_id]
        elif class_id not in img_idx_dict[image_id]:
            img_idx_dict[image_id] += [class_id]
        else:
            continue
        
    # select images based on whether not the image has a segmentation of the target object
    pos_data = []
    neg_data = []
    for imgc in img_idx_dict.keys():
        if target in img_idx_dict[imgc]:
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
    ltr_path = os.path.join(COCO_STUFF_DIR, "annotations/instances_train2017.json")
    lt_path = os.path.join(COCO_STUFF_DIR, "annotations/instances_val2017.json")
    train_path = os.path.join(COCO_STUFF_DIR, "train2017")
    test_path = os.path.join(COCO_STUFF_DIR, "val2017")

    train_coco = load_coco_data(train_path, ltr_path, target=41, n_samples=500)
    test_coco = load_coco_data(test_path, lt_path, target=41, n_samples=250)

    first_x, first_y = next(iter(test_coco))
    
    print(first_x)
    print(first_y)
