"""Creat the folder with the right classes in data folder."""

import os
from glob import glob
import shutil
import pandas as pd

BASE_FOLDER = "data/skin-cancer-mnist-ham10000/versions/2/"
IMAGE_FOLDERS = [
    "data/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1",
    "data/skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2"
]

LESION_TYPE_DICT = {
    'mel': 'Melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma',
    'nv': 'melanocytic_nevi'
}

DESTINATION_FOLDER = "data/"


def create_data_folder(lesion_type_dict: str):
    """Put images into the folder that represent its class.

    Args:
        lesion_type_dict (str): Ditionary with the abbreviated lesion type as key and the complete lesion type as value.
    """
    # Read the metadata file
    data = pd.read_csv(os.path.join(BASE_FOLDER, "HAM10000_metadata.csv"))
    data["image_id"] = data["image_id"].apply(lambda x: x + ".jpg")

    for img in glob(BASE_FOLDER + "**/*.jpg"): 
        lesion_type_small = data[data["image_id"] == img.split("/")[-1]]["dx"].values[0]
        destination = f"data/images/{lesion_type_dict[lesion_type_small]}"

        if not os.path.exists(destination):
            os.makedirs(destination)
        
        shutil.copy(img, destination)


def split_data(train_share: float, test_share: float):
    """_summary_

    Args:
        train_share (float): _description_
        test_share (float): _description_
    """
    assert train_share + test_share <= 1, "Train and test share must be less than 1"
    assert train_share >= 0, "Train share must be greater than or equal to 0"
    assert test_share >= 0, "Test share must be greater than or equal to 0"

    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    if not os.path.exists("data/test"):
        os.makedirs("data/test")
    if not os.path.exists("data/valid"):
        os.makedirs("data/valid")

    base_dir = "data/images/"

    for folder in os.listdir(base_dir):
        print(folder)

        images = glob(os.path.join(base_dir, folder, "*.jpg"))
        
        train_data = images[:int(len(images) * train_share)]
        test_data = images[
            int(len(images) * train_share): int(len(images) * (train_share+test_share))
        ]
        valid_data = images[
            int(len(images) * (train_share + test_share)):]
        
        for img in train_data:
            destination = f"data/train/{folder}"
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(img, destination)

        for img in test_data:
            destination = f"data/test/{folder}"
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(img, destination)

        for img in valid_data:
            destination = f"data/valid/{folder}"
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(img, destination)

if __name__ == "__main__":
    create_data_folder(
        image_folders=IMAGE_FOLDERS,
        lesion_type_dict=LESION_TYPE_DICT
    )
    split_data(
        train_share=0.7,
        test_share=0.1
    )