"""Making the data loaders from the data sets.

We will generate the data loaders for the train, test and validation sets.
"""

import os
from glob import glob
from typing import List, Tuple
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

# TODO: add logging

def data_stats(
    data_dir: str,
    img_size: Tuple[int]=(224, 224),
):
    """Find the meand and standard deviation of the dataset.

    Args:
        data_dir (str): directory to the train data. example: "data/train"
        img_size (Tuple[int], optional): The images will be resized. Defaults to (224, 224).
    """

    img_height, img_width = img_size
    imgs = []
    means, stdevs = [], []

    
    for image in tqdm(glob(f"{data_dir}/**/*.jpg")):
        img = Image.open(image)
        img = img.resize((img_width, img_height))
        imgs.append(img)
    
    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
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
    transform_list = []

    if resize:
        transform_list.append(transforms.Resize(resize))
    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
    # Normalize the image
    transform_list.append(
        transforms.Normalize(mean=normalize[0], std=normalize[1])
    )
    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    train_transform = transforms.Compose(transform_list)
    valid_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ]
    )
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

    train_data_path = f"{data_dir}/train"
    test_data_path = f"{data_dir}/test"
    valid_data_path = f"{data_dir}/valid"

    train_transform, valid_transform = transforms

    train_data_loader = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform,
        shuffle=shuffle
    )

    valid_data_loader = torchvision.datasets.ImageFolder(
        root=valid_data_path,
        transform=valid_transform,
        shuffle=shuffle
    )

    return train_data_loader, valid_data_loader


if __name__ == "__main__":
    mean, stdev = data_stats(
        data_dir="data/train",
        img_size=(224, 224)
    )
    train_transform, valid_transform = create_transform(
        resize=(244, 244),
        normalize=(mean, stdev),
        random_horizontal_flip=True,
        random_vertical_flip=True,
    )
    train_data_loader, valid_data_loader = prepare_data(
        data_dir="data",
        batch_size=32,
        shuffle=True,
        transforms=(train_transform, valid_transform)
    )
    # TODO: Check some of timages in the train_data_loader

