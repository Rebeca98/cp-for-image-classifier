import torchvision.models as models
import torch.nn as nn
import torch
from utils import get_model
from paths import *
import os

# constants
IMAGE_SIZE = 580


def build_model(modelname='efficientnet_b0', num_classes=14, pretrained=True, fine_tune=True, model_name="model.pth"):
    """
    build model for training
    """
    if pretrained == False:
        # we load our trained weights
        model_path = os.path.join(MODEL_PATH, model_name)
        pretrained_weights = torch.load(model_path)
        model = get_model(modelname, pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes)
        model.load_state_dict(pretrained_weights['model_state_dict'])
    else:
        model = get_model(modelname, pretrained)
    # Change the final classification head.
        num_ftrs = model.classifier[1].in_features
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes)
    return model


def build_model_inference(modelname='efficientnet_b0', num_classes=14, pretrained=True, model_name="model.pth"):
    """
    build model for training
    """
    model_path = os.path.join(MODEL_PATH, model_name)
    pretrained_weights = torch.load(model_path)  # we load our trained weights
    model = get_model(modelname, pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=num_ftrs, out_features=num_classes)
    model.load_state_dict(pretrained_weights['model_state_dict'])
    return model
