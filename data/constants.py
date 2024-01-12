import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/ 
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "artifacts/data/CUB_200_2011"
CUB_PROCESSED_DIR = "artifacts/data/class_attr_data_10"

# COCO-Stuff Constants
# COCO-Stuff data is obtained from : https://github.com/nightrome/cocostuff
COCO_STUFF_DIR = "artifacts/data/COCO_STUFF"

# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "/path/to/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "/path/to/broden/"

# SIIM-ISIC can be obtained from : https://drive.google.com/file/d/1H7cTkuEF2vg7UsiLjnFKJ7kBPIN1Y3IL/view?usp=drive_link
# The original full dataset can be found here : https://challenge2020.isic-archive.com/
SIIM_DATA_DIR = "artifacts/data/SIIM_ISIC"

# BRODEN concept bank
BRODEN_CONCEPTS = "artifacts/data/broden_concepts"