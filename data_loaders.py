"""Making the data loaders from the data sets.

We will generate the data loaders for the train, test and validation sets.
"""

import os
import logging
from glob import glob
from typing import List, Tuple
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def data_stats(
    data_dir: str,
    img_size: Tuple[int]=(224, 224),
):
    """Find the meand and standard deviation of the dataset.

    Args:
        data_dir (str): directory to the train data. example: "data/train"
        img_size (Tuple[int], optional): The images will be resized. Defaults to (224, 224).
    """

    logger.info("---> Starting Image Stats")

    img_height, img_width = img_size
    imgs = []
    means, stdevs = [], []

    for image in tqdm(glob(f"{data_dir}/**/*.jpg")):
        img = Image.open(image)
        img = img.resize((img_width, img_height))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    logger.info("normMean = {}".format(means))
    logger.info("normStd = {}".format(stdevs))
    logger.info("<--- Exiting Image Stats. Successfully outputing image stats.")
    return means, stdevs


def create_transform(
    resize: List,
    normalize: List,
    random_horizontal_flip: bool=True,
    random_vertical_flip: bool=True,
) -> transforms:
    """Create a transformer for the dataset.

    Args:
        resize (List): _description_
        normalize (List): _description_
        random_horizontal_flip (bool, optional): _description_. Defaults to True.

    Returns:
        transforms: _description_
    """

    logger.info("---> Starting Create Transformer.")
    transform_list = []

    if resize is not None:
        transform_list.append(transforms.Resize((224, 224)))
    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
    # Normalize the image
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize[0], std=normalize[1])
    ])

    train_transform = transforms.Compose(transform_list)
    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ]
    )
    logger.info("<--- Exiting Create Transformer. Transformers created successfully.")
    return train_transform, valid_transform


def prepare_data(
    data_dir: str,
    batch_size: int=32,
    shuffle: bool=True,
    transforms: Tuple=None
) -> List[DataLoader]:

    """_summary_

    Args:
        data_dir (str): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        List[DataLoader]: _description_
    """
    logger.info("---> Starting Prepare Data.")

    train_data_path = f"/home/bigbang/workshop/projects/skin-cancer-tool/data/train"
    test_data_path = f"/home/bigbang/workshop/projects/skin-cancer-tool/data/test"
    valid_data_path = f"/home/bigbang/workshop/projects/skin-cancer-tool/data/valid"

    train_transform, valid_transform = transforms

    logger.info("Creating Train Data Loader.")
    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=shuffle
    )

    logger.info("Creating Valid Data Loader.")
    valid_data = torchvision.datasets.ImageFolder(
        root=valid_data_path,
        transform=valid_transform
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=shuffle
    )

    logger.info("<--- Exiting Prepare Data. Successfully created dataloaders.")
    return train_data_loader, valid_data_loader

# TODO: Check some of Images in the train_data_loader
# TODO: Add Test DataLoader
