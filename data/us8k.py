import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from .constants import US_DIR


class US8KDataset(Dataset):

    def __init__(self, datalist, transform=None, sample_rate:int=44100, max_len:int=176400):
        """
        Arguments:
        datalist: a list object instance returned by 'load_esc_data' function below
        transform: whether to apply any special transformation. Default = None
        sample_rate: the sampling rate of the audio data loaded (default is general standard)
        max_len: due to the variations in audio file lengths, this is needed to ensure that
        the dataloader contains only samples of the same length
        This variable will then be used to model the target size of the outputs
        """
        self.data = datalist
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_data = self.data[idx]
        audio_path = audio_data[0]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, dtype=np.float32)
        audio = audio.reshape(1, -1)
        audio = torch.from_numpy(audio)

        class_label = audio_data[1]
        if self.transform:
            audio = self.transform(audio)
        
        delta = [0, self.max_len - audio.size(dim=-1)]
        audio = pad(audio, delta)

        return audio, class_label

# Function for preparing the data from the data (2000 files split following the author's recommendations for one split)
# Here, folds 1-4 are used for training while fold 5 is used for test by default
def prepare_data(df, testfolds:list=[9, 10]):
    """
    Arguments:
    df: Pandas dataframe object
    testfold: fold(s) used for testing (the rest would then be used for training)

    Outputs:
    train_data, test_data: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    train_df = df[-df["fold"].isin(testfolds)]
    test_df = df[df["fold"].isin(testfolds)]

    X_train, y_train = list(train_df['filename']), list(train_df['classID'])
    X_test, y_test = list(test_df['filename']), list(test_df['classID'])

    train_data = [[X_train[idx], y_train[idx]] for idx in range(len(X_train))]
    test_data = [[X_test[idx], y_test[idx]] for idx in range(len(X_test))]

    return train_data, test_data


def load_us_data(meta_dir, transform=None, batch_size:int=1, testfolds:list=[9, 10]):
    """
    Arguments:
    meta_dir: path to CSV file containing all metadata
    transform: transformations to use
    batch_size: number of datapoints in a batch
    testfold: fold(s) used for testing (the rest would then be used for training, see prepare_data above for more details)

    Outputs:
    train_loader, test_loader: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    df = pd.read_csv(meta_dir).drop(['fsID', 'start', 'end', 'salience'], axis=1)
    train_dir = os.path.join(US_DIR, "fold")
    df['filename'] = train_dir + df['fold'].astype(str) + "/" + df['slice_file_name']
    train_data, test_data = prepare_data(df, testfolds)

    train_data = US8KDataset(train_data, transform)
    test_data = US8KDataset(test_data, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

# For testing:
if __name__ == "__main__":

    meta_path = os.path.join(US_DIR, "UrbanSound8K.csv")
    train_esc, test_esc = load_us_data(meta_path)

    first_x, first_y = next(iter(train_esc))
    
    print(first_x.size())
    print(first_y)
    