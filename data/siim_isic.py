import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from .constants import SIIM_DATA_DIR

class ISICDataset(Dataset):

    def __init__(self, datalist, transform=None, cropsize=(480, 640)):
        """
        Arguments:
        annot_dir: directory of the corresponding annotation file
        datalist: a list object instance returned by 'data_select' function above
        transform: whether to apply any special transformation. Default = None
        """
        self.data = datalist
        self.transform = transform
        self.cropsize = cropsize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_data = self.data[idx]
        img_path = img_data[0]
        img = Image.open(img_path).convert('RGB')
        img = transforms.CenterCrop(self.cropsize)(img)
        img = transforms.ToTensor()(img)

        class_label = img_data[1]
        if self.transform:
            img = self.transform(img)

        return img, class_label

# Function for preparing the data from the data (2500 in total which have been pre-selected: 80% benign, 20% malignant)
def prepare_data(df, ratio:float=0.2, seed:int=42):
    """
    Arguments:
    df: Pandas dataframe object
    ratio: percentage of data to use for testing
    seed: random seed for train test split

    Outputs:
    train_data, test_data: Lists in the form of a PyTorch datalist; '[[img_path1, img_label1], [img_path2, img_label2], ...]'
    """
    datalist = [list(df['image_name']), list(df['target'])] # PyTorch data list in the format of '[[img_paths], [img_labels]]'
    X_train, X_test, y_train, y_test = train_test_split(datalist[0], datalist[1], 
                                                        stratify=datalist[1], test_size=ratio, 
                                                        random_state=seed)
    
    train_data = [[X_train[idx], y_train[idx]] for idx in range(len(X_train))]
    test_data = [[X_test[idx], y_test[idx]] for idx in range(len(X_test))]

    return train_data, test_data


def load_siim_data(meta_dir, transform=None, batch_size:int=1, seed:int=42):
    """
    Arguments:
    meta_dir: path to CSV file containing all metadata
    transform: transformations to use
    batch_size: number of datapoints in a batch
    seed: random seed for train test split

    Outputs:
    train_loader, test_loader: Lists in the form of a PyTorch datalist; '[[img_path1, img_label1], [img_path2, img_label2], ...]'
    """
    df = pd.read_csv(meta_dir)[['image_name', 'target']]
    train_dir = os.path.join(SIIM_DATA_DIR, "data")
    df['image_name'] = train_dir + "/" + df['image_name'] + ".jpg"
    train_data, test_data = prepare_data(df, seed)

    train_data = ISICDataset(train_data, transform)
    test_data = ISICDataset(test_data, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

# For testing:
if __name__ == "__main__":

    meta_path = os.path.join(SIIM_DATA_DIR, "isic_metadata.csv")
    train_siim, test_siim = load_siim_data(meta_path)

    first_x, first_y = next(iter(train_siim))
    
    print(first_x)
    print(first_y)