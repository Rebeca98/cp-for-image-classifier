import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import numpy as np
from paths import *

# Required constants.
VALID_SPLIT = 0.1  # 10% of training data
IMAGE_SIZE = 580  # Image size of resize when applying transforms. #probar con 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Number of parallel processes for data preparation.


class MyEqualizerTransform:
    """Rotate by one of the given angles."""

    def __init__(self, arg1=None):
        self.__arg1 = arg1

    def __call__(self, x):
        return TF.equalize(x)

# Train transforms


def get_train_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        MyEqualizerTransform(),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ColorJitter(
            brightness=.3, contrast=0.5, saturation=0.59, hue=.3),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=[
            #transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ], p=1)
    ])
    return transform

# Validation transforms


def get_valid_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=[
            #transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ], p=1)
    ])
    return transform


def get_model_data_loaders(train_sampler, valid_sampler, batch_size):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """

    dataset_train = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_train_transform())
    )
    dataset_valid = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_valid_transform())
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, sampler=valid_sampler, num_workers=NUM_WORKERS)

    # train_loader = DataLoader(
    #    dataset_train, batch_size=BATCH_SIZE,
    #    shuffle=True, num_workers=NUM_WORKERS
    # )
    # valid_loader = DataLoader(
    #    dataset_valid, batch_size=BATCH_SIZE,
    #    shuffle=False, num_workers=NUM_WORKERS
    # )
    return train_loader, valid_loader
