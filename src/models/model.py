import torchvision.models as models
import torch.nn as nn
import torch
from utils import get_model
from paths import *

def build_model(modelname='efficientnet_b0', num_classes=14, pretrained=True):
    """
    build model for training
    """
    if pretrained == False:
        # we load our trained weights
        pretrained_weights = torch.load(MODEL_PATH)
        model = get_model(modelname, pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes)
        model.load_state_dict(pretrained_weights['model_state_dict'])
    else:
        model = get_model(modelname, pretrained)
    # Change the final classification head.
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes)
    for params in model.parameters():
        params.requires_grad = True
    return model


def build_model_inference(modelname='efficientnet_b0', num_classes=14, pretrained=True):
    """
    build model for training
    """
    pretrained_weights = torch.load(MODEL_PATH)  # we load our trained weights
    model = get_model(modelname, pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=num_ftrs, out_features=num_classes)
    model.load_state_dict(pretrained_weights['model_state_dict'])
    return model
