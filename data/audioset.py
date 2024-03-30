import os
import csv
import pandas as pd
import torch
import librosa
import numpy as np
import yt_dlp
from re import sub
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .constants import AS_DIR, AS_TRAIN_IDX, AS_EVAL_IDX


class AudioSetDataset(Dataset):

    def __init__(self, datalist, transform=None, sample_rate: int = 44100):
        """
        Arguments:
        datalist: a list object instance returned by 'load_aud_data' function below
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


# Function for reformatting the csv segment files due to their unorthodox formatting
# (and how we cannot rewrite only one row)
def reformat_csv(csv_file: str):
    """
    Arguments:
    csv_file: path to the index file ("balanced_train_segments.csv"/"eval_segments.csv")
    """
    base_data = ["#YTID", "start_seconds", "end_seconds"]
    pos_labels = [f"positive_labels_{n}" for n in range(9)]
    new_row_data = base_data + pos_labels  # New data for the row
    check_labels = base_data + ["positive_labels"]

    # Read the base CSV file
    with open(csv_file, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Rewrite if it has not been already
    if rows[0] == check_labels:
        print("Dataset already re-processed. Skipping...")
        return

    else:
        rows[2] = new_row_data

        # Write the modified CSV file
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        # Combine the labels into one column
        df = pd.read_csv(csv_file, on_bad_lines="skip", skiprows=2)
        df["positive_labels"] = df.iloc[:, 3:].agg(
            lambda row: ", ".join(row.dropna()), axis=1
        )

        # Convert #YTID to URL
        df["#YTID"] = "https://www.youtube.com/watch?v=" + df["#YTID"]

        # Apostrophe removal
        df["positive_labels"] = df["positive_labels"].str.replace('"', "")
        df = df.drop(columns=pos_labels)

        df.to_csv(csv_file, index=False)


# Function for downloading and compiling the audio segments
def download_data(df, data: str = "train", n_data_points: int = 50):
    """
    Arguments:
    df: Pandas dataframe object corresponding to either the training or evaluation set
    data: If the data is either "train" or "test" (used for file saving)
    n_data_points: Number of datapoints to return per class

    Outputs:
    data: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    # Creating the directory
    split_dir = os.path.join(AS_DIR, data)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    # For smarter file tracking
    archive = os.path.join(AS_DIR, f"archive_{data}.txt")
    if not Path(archive).exists():
        with open(archive, "w") as file:
            file.write("")

    # Get the classes and indices
    class_path = os.path.join(AS_DIR, "class_labels_indices.csv")
    cls = pd.read_csv(class_path)
    idxs, mids = list(cls["index"]), list(cls["mid"])
    mid2idx = {mids[n]: idxs[n] for n in range(len(idxs))}
    midcounts = {mid: 0 for mid in mids}

    # Download the files
    datalist = []
    for _, entry in df.iterrows():
        # Getting the details
        url = entry.iloc[0]
        start, end = int(entry.iloc[1]), int(entry.iloc[2])
        pos_labels = sub(" ", "", entry.iloc[3]).split(",")
        vid_id = url.split("?v=")[1]
        filepath = os.path.join(split_dir, vid_id)

        # Download only if not present in target folder
        ydl_opts = {
            "download_archive": archive,
            "format": "bestaudio/best",
            "ffmpeg-location": "C:\\ffmpeg\\bin\\ffmpeg.exe",  # Note: this *may* need to be adjusted depending on the device
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
            "download_ranges": yt_dlp.utils.download_range_func(
                None, [(start, end)]
            ),  # Defining the range to download
            "outtmpl": filepath,
            "force-keyframes-at-cuts": True,
            "prefer-ffmpeg": True,
            "ignoreerrors": True,  # Continue even if we get an error
        }

        # Try to download the snippet if it's needed
        if any([midcounts[label] < n_data_points for label in pos_labels]):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download(url)

        # Ignore the video if it returns an error
        if error_code:
            continue

        # Else make sure that it doesn't exceed the count (n_data_points)
        else:
            filepath += ".m4a"
            for label in pos_labels:
                if midcounts[label] >= n_data_points:
                    continue

                else:
                    midcounts[label] += 1
                    datalist.append([filepath, mid2idx[label]])

        # Check if we have n_data_points items to stop the operations
        if all(i >= n_data_points for i in list(midcounts.values())):
            break

    # Saving the data (to prevent having to iterate over the entries again)
    df = pd.DataFrame(datalist, columns=["path", "index"])
    data_path = os.path.join(AS_DIR, f"{data}_data_{n_data_points}.csv")
    df.to_csv(data_path, index=False)
    print(f"Data has been saved to {data_path}!")

    return datalist


# Function for processing the data if already present (i.e., from validation)
def process_data(
    df, data: str = "valid", extension: str = "wav", n_data_points: int = 50
):
    """
    Arguments:
    df: Pandas dataframe object corresponding to either the training or evaluation set
    data: If the data is either "train" or "valid" (used for file saving)
    extension: The extension of the files present (default = "wav")
    n_data_points: Number of datapoints to return per class

    Outputs:
    data: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    # Creating the directory
    split_dir = os.path.join(AS_DIR, data)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    # Get the classes and indices
    class_path = os.path.join(AS_DIR, "class_labels_indices.csv")
    cls = pd.read_csv(class_path)
    idxs, mids = list(cls["index"]), list(cls["mid"])
    mid2idx = {mids[n]: idxs[n] for n in range(len(idxs))}
    midcounts = {mid: 0 for mid in mids}

    # Download the files
    datalist = []
    for _, entry in df.iterrows():
        # Getting the details
        url = entry.iloc[0]
        pos_labels = sub(" ", "", entry.iloc[3]).split(",")
        vid_id = url.split("?v=")[1]
        filepath = os.path.join(split_dir, vid_id)
        filepath += f".{extension}"

        # Make sure that the datapoints present don't exceed the count (n_data_points)
        for label in pos_labels:
            if midcounts[label] >= n_data_points:
                continue

            else:
                midcounts[label] += 1
                datalist.append([filepath, mid2idx[label]])

            # Check if we have n_data_points items to stop the operations
            if all(i >= n_data_points for i in list(midcounts.values())):
                break

    # Saving the data (to prevent having to iterate over the entries again)
    df = pd.DataFrame(datalist, columns=["path", "index"])
    data_path = os.path.join(AS_DIR, f"{data}_data_{n_data_points}.csv")
    df.to_csv(data_path, index=False)
    print(f"Data has been saved to {data_path}!")

    return datalist


# Function for retrieving the data (or commencing the downloads if the files are not present)
def get_data(df, data: str = "valid", extension: str = "wav", n_data_points: int = 50):
    """
    Arguments:
    df: Pandas dataframe object corresponding to either the training or evaluation set
    data: If the data is either "train" or "test" (used for file saving)
    extension: The extension of the files present (default = "wav")
    n_data_points: Number of datapoints to return per class

    Outputs:
    data: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    # Initial preparation
    data_path = os.path.join(AS_DIR, f"{data}_data_{n_data_points}.csv")
    split_dir = os.path.join(AS_DIR, data)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    # If the data mapping file already exists, then just use it
    if os.path.exists(data_path):
        print("CSV data file detected. Loading...")
        datalist = pd.read_csv(data_path).values.tolist()

    # Else if the files are there (i.e., from validation), we process that
    elif len(os.listdir(split_dir)) > 0:
        print("Files detected. Generating dataset csv...")
        datalist = process_data(
            df, data, extension=extension, n_data_points=n_data_points
        )

    # Otherwise we need to compile the data from scratch
    else:
        print("Files not found. Downloading data...")
        datalist = download_data(df, data, n_data_points)

    return datalist


def load_aud_data(
    data: str = "valid",
    transform=None,
    batch_size: int = 1,
    num_workers: int = 4,
    n_data_points: int = 50,
    val_size: float = 0.2,
):
    """
    Arguments:
    split: which split to use (default = "valid")
    transform: transformations to use
    batch_size: number of datapoints in a batch
    num_workers: the amount of workers to use for the torch dataloaders created
    n_data_points: number of datapoints to return per class
    val_size: the size of the validation data (has to be between 0.0 and 1.0)

    Outputs:
    train_loader, test_loader: Lists in the form of a PyTorch datalist; '[[audio_path1, audio_label1], [audio_path2, audio_label2], ...]'
    """
    # Initial checks
    data = data.lower()
    assert data in [
        "train",
        "valid",
    ], "Please enter 'train' or 'valid' for the 'data' parameter!"

    curr = os.getcwd()
    csv_file = (
        os.path.join(curr, AS_TRAIN_IDX)
        if data == "train"
        else os.path.join(curr, AS_EVAL_IDX)
    )

    assert (val_size >= 0) and (
        val_size <= 1
    ), "Please enter a 'val_size' between 0.0 and 1.0!"

    # First ensure that the csv files are reformatted (in case the non-validation files are desired)
    reformat_csv(csv_file)
    df = pd.read_csv(csv_file)
    data = get_data(df, data=data, n_data_points=n_data_points)

    # And now we split the data in stratified fashion
    # Split data into features (X) and labels (y)
    X, y = [item[0] for item in data], [item[1] for item in data]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )

    # Now we convert the lists into dataloaders
    train_data, test_data = list(zip(X_train, y_train)), list(zip(X_test, y_test))
    train_dataset = AudioSetDataset(train_data, transform)
    test_dataset = AudioSetDataset(test_data, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, test_loader


# For testing:
if __name__ == "__main__":

    train_as, test_as = load_aud_data(num_workers=2)

    first_x, first_y = next(iter(train_as))

    print(first_x)
    print(first_y)
