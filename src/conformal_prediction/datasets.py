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
VALID_SPLIT = 0.1 #10% for validation 
IMAGE_SIZE = 580 # Image size of resize when applying transforms. #probar con 224
BATCH_SIZE = 32 
NUM_WORKERS = 0 # Number of parallel processes for data preparation.
NUM_CALIB = 100 # esto debe de calcularse mejor como una proporcion

class MyEqualizerTransform:
    """Rotate by one of the given angles."""

    def __init__(self, arg1=None): 
        self.__arg1 = arg1

    def __call__(self, x):
        return TF.equalize(x)

# Training trasnforms
def get_train_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        MyEqualizerTransform(),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ColorJitter(brightness=.3,contrast=0.5,saturation=0.59,hue=.3),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=
                               [
                                #transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                               ], p=1)
            ])
    return transform

# Validation transforms
def get_valid_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=
                               [
                                #transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                               ], p=1)
        ])
    return transform

def get_calib_transform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=[transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])], p=0.6)])
    return transform

def get_model_dataloaders(batch_size):
    """
    Function to prepare the Datasets.

    :param pretrained: Boolean, True or False.

    Returns the training and validation datasets along 
    with the class names.
    """
    train_dataset = datasets.ImageFolder(
        TRAIN_PATH,
        transform=(get_train_transform())
    )
    test_dataset = datasets.ImageFolder(
        TEST_PATH,
        transform=(get_train_transform())
    )
    print(f"[INFO]: Number of training images: {len( train_dataset)}")
    print(f"[INFO]: Number of test images: {len(test_dataset)}")
    
    # dataloaders
    train_loader = DataLoader(train_dataset,
        batch_size=batch_size,shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,shuffle=True
    )

   
    #train_dataset_size = len(train_dataset)
    #targets = train_dataset.targets

    # Calculate the validation dataset size.
    #valid_size = int(VALID_SPLIT*dataset_size) #10%
    #dataset_train, dataset_valid = torch.utils.data.random_split(dataset,[dataset_size-valid_size,valid_size])
    
    #train_idx, valid_idx= train_test_split(
    #    np.arange(len(targets)),
    #    test_size=VALID_SPLIT,
    #    shuffle=True,
    #    stratify=targets)
    
    #train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    #valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    
    
    # Radomize the data indices.
    #indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    #dataset_train = Subset(dataset, indices[:-valid_size])
    #dataset_valid = Subset(dataset_test, indices[-valid_size:])
    #num_classes = len(dataset.classes)
    #return dataset_train, dataset_valid,dataset.classes,dataset
    return train_loader, test_loader,train_dataset.classes

def get_cmodel_dataset():
    """
    Function to prepare the Datasets.

    :param pretrained: Boolean, True or False.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_calibration = datasets.ImageFolder(
        CALIB_PATH, 
        transform=(get_calib_transform())
    )
    dataset_calibration_size = len(dataset_calibration)
    valid_size = int(VALID_SPLIT*dataset_calibration_size) 
    calib_data, calib_val_data = torch.utils.data.random_split(dataset_calibration,[dataset_calibration_size-valid_size,valid_size])
    
    #num_classes = len(dataset.classes)

    return calib_data, calib_val_data

# datasets 
def get_model_data_loaders(train_sampler, valid_sampler,batch_size):
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
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler,num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=valid_sampler,num_workers=NUM_WORKERS)
    
    #train_loader = DataLoader(
    #    dataset_train, batch_size=BATCH_SIZE, 
    #    shuffle=True, num_workers=NUM_WORKERS
    #)
    #valid_loader = DataLoader(
    #    dataset_valid, batch_size=BATCH_SIZE, 
    #    shuffle=False, num_workers=NUM_WORKERS
    #)
    return train_loader,valid_loader
    

def get_cmodel_data_loaders(calib_data, calib_val_data):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    calib_loader = DataLoader(
        calib_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
        )
    val_loader = DataLoader(
        calib_val_data, batch_size= BATCH_SIZE, 
        shuffle= True, num_workers=NUM_WORKERS
        )    
    return calib_loader , val_loader