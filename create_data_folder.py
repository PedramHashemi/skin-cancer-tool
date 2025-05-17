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

# Read the metadata file
data = pd.read_csv(os.path.join(BASE_FOLDER, "HAM10000_metadata.csv"))
data["image_id"] = data["image_id"].apply(lambda x: x + ".jpg")


def create_data_folder(lesion_type_dict: str):
    """Put images into the folder that represent its class.

    Args:
        lesion_type_dict (str): Ditionary with the abbreviated lesion type as key and the complete lesion type as value.
    """
    for img in glob(BASE_FOLDER + "**/*.jpg"): 
        lesion_type_small = data[data["image_id"] == img.split("/")[-1]]["dx"].values[0]
        destination = f"data/images/{lesion_type_dict[lesion_type_small]}"

        if not os.path.exists(destination):
            os.makedirs(destination)
        
        shutil.copy(img, destination)

def split_data(train_share, test_share):


if __name__ == "__main__":
    create_data_folder(
        image_folders=IMAGE_FOLDERS,
        lesion_type_dict=LESION_TYPE_DICT
    )