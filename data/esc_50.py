import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .constants import ESC_DIR

class ESCDataset(Dataset):

    def __init__(self, datalist, transform=None, sample_rate:int=44100):
        """
        Arguments:
        datalist: a list object instance returned by 'load_esc_data' function below
        transform: whether to apply any special transformation. Default = None
        sample_rate: the sampling rate of the audio data loaded (default is general standard)
        """
        self.data = datalist
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_data = self.data[idx]
        audio_path = audio_data[0]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, dtype=np.float32)
        audio = audio.reshape(1, -1)

        class_label = audio_data[1]
        if self.transform:
            audio = self.transform(audio)

        return audio, class_label

# Function for preparing the data from the data (2000 files split following the author's recommendations for one split)
# Here, folds 1-4 are used for training while fold 5 is used for test by default
def prepare_data(df, testfold:int=1):
    """
    Arguments:
    df: Pandas dataframe object
    testfold: fold used for testing (the rest would then be used for training)

    Outputs:
    train_data, test_data: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    train_df = df[df["fold"] != testfold]
    test_df = df[df["fold"] == testfold]

    X_train, y_train = list(train_df['filename']), list(train_df['target'])
    X_test, y_test = list(test_df['filename']), list(test_df['target'])

    train_data = [[X_train[idx], y_train[idx]] for idx in range(len(X_train))]
    test_data = [[X_test[idx], y_test[idx]] for idx in range(len(X_test))]

    return train_data, test_data


def load_esc_data(meta_dir, transform=None, batch_size:int=1, testfold:int=5, num_workers:int=4):
    """
    Arguments:
    meta_dir: path to CSV file containing all metadata
    transform: transformations to use
    batch_size: number of datapoints in a batch
    testfold: fold used for testing (the rest would then be used for training, see prepare_data above for more details)
    num_workers: the amount of workers to use for the torch dataloaders created

    Outputs:
    train_loader, test_loader: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    df = pd.read_csv(meta_dir).drop(['src_file', 'take'], axis=1)
    train_dir = os.path.join(ESC_DIR, "audio")
    df['filename'] = train_dir + "/" + df['filename']
    train_data, test_data = prepare_data(df, testfold)

    train_data = ESCDataset(train_data, transform)
    test_data = ESCDataset(test_data, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader

# For testing:
if __name__ == "__main__":

    meta_path = os.path.join(ESC_DIR, "esc50.csv")
    train_esc, test_esc = load_esc_data(meta_path, num_workers=2)

    first_x, first_y = next(iter(train_esc))
    
    print(first_x)
    print(first_y)