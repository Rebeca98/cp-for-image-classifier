import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset


class MyEqualizerTransform:
    """Rotate by one of the given angles."""

    def __init__(self, arg1=None):
        self.__arg1 = arg1

    def __call__(self, x):
        return TF.equalize(x)

# Training trasnformations
def get_train_transform(image_size):
    """
    
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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

# This function does not work here
def get_valid_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=[
            #transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ], p=1)
    ])
    return transform

def get_validation_transform(image_size):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    return transform

# This function does not work here
def get_calib_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(transforms=[transforms.Normalize([0.3768, 0.3809, 0.3522], [0.1951, 0.1968, 0.1943])], p=0.6)])
    return transform


def get_model_dataloaders(batch_size,train_dir,validation_dir,image_size):
    """
    Function that return dataloaders for train and validation sets.

    batch_size: int
    train_dir: str
        directory for training data (images)
    test_dir: str
        directory for testing data (images)
    image_size: int
        image size (height=width)

    Returns the training and validation datasets along 
    with the class names.
    """
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=(get_train_transform(image_size))
    )
    test_dataset = datasets.ImageFolder(
        validation_dir,
        transform=(get_train_transform(image_size))
    )
    print(f"[INFO]: Number of training images: {len( train_dataset)}")
    print(f"[INFO]: Number of test images: {len(test_dataset)}")

    # dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True
                              )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True
    )
    return train_loader, test_loader, train_dataset.classes


def get_model_dataloaders_evaluation(batch_size,test_dir,image_size,**kwargs):
    """
    Function that return dataloaders for train and validation sets.

    batch_size: int
    train_dir: str
        directory for training data (images)
    test_dir: str
        directory for testing data (images)
    image_size: int
        image size (height=width)

    Returns the training and validation datasets along 
    with the class names.
    """
    validation_dataset = datasets.ImageFolder(
        test_dir,
        transform=(get_validation_transform(image_size))
    )
    subset = kwargs.get('subset', False)
    if subset:
        import random
        subset_indices = random.sample(range(len(validation_dataset)), 2000)
        #subset_indices = list(range(100))  # Indices 0 to 99 (inclusive)
        subset = Subset(validation_dataset, subset_indices)
        validation_loader = DataLoader(subset,
                              batch_size=batch_size, shuffle=True
                              )
        return validation_loader, validation_dataset.classes
    
    print(f"[INFO]: Number of validation images: {len(test_dir)}")

    # dataloaders
    validation_loader = DataLoader(validation_dataset,
                              batch_size=batch_size, shuffle=True
                              )

    return validation_loader, validation_dataset.classes



