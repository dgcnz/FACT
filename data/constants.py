import os
from pathlib import Path

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "artifacts/data/CUB_200_2011"
CUB_PROCESSED_DIR = "artifacts/data/class_attr_data_10"

# COCO-Stuff Constants
# COCO-Stuff data is obtained from : https://github.com/nightrome/cocostuff
# The dataset used in this project is from a Hugging Face repository (link can be found in the download script)
COCO_STUFF_DIR = "artifacts/data/COCO_STUFF"

# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html

DERM7_FOLDER = "artifacts/data/derm7pt"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "artifacts/data/HAM10K"

# The dataset used in this project is from a Hugging Face repository (link can be found in the download script)
# The original full dataset can be found here : https://challenge2020.isic-archive.com/
SIIM_DATA_DIR = "artifacts/data/SIIM_ISIC"

# BRODEN concept bank
BRODEN_CONCEPTS = "artifacts/data/broden_concepts"


# CONSTANTS FOR EXTENSION EXPERIMENTS
# ESC_50 can be obtained from : https://github.com/karolpiczak/ESC-50
ESC_DIR = "artifacts/data/ESC_50"
ESC_CONCEPTS = Path("configs") / "conceptnet" / "esc50" / "output_v0.yaml" # can be changed

# UrbanSound8k can be obtained from : https://www.kaggle.com/datasets/chrisfilo/urbansound8k
US_DIR = "artifacts/data/US8K"
US_CONCEPTS = Path("configs") / "conceptnet" / "us8k" / "output_v1.yaml" # can be changed

# AudioSet can be obtained from : https://research.google.com/audioset/download.html
AS_DIR = "artifacts/data/audioset"
AS_TRAIN_IDX = os.path.join(AS_DIR, "balanced_train_segments.csv")
AS_EVAL_IDX = os.path.join(AS_DIR, "eval_segments.csv")
