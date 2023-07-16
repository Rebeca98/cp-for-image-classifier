#import torchvision.models as models
import torch.nn as nn
import torch
from classification_model.model.utils import get_model
import os


def build_model(model_weights_path, 
                architecture,
                model_name, 
                num_classes, 
                pretrained, 
                fine_tune, 
                trained_model_name=None):
    """
    Build model for training.
    if pretrained is true then with this function you can continue 
    training your model with your last trained weights, if not you can build your model to start trainining. 
    Also you have fine tunning option.

    Args:
        model_weights_path (str): Path for the model weights to be saved to.
        architecture (str): choose an architecture (e.g. efficientnet_b0) to make our transfer learning / fine tunning   
        num_classes (int): number of classes (e.g. 14)
        pretrained (bool): if True it indicates that we want to continue training our model with our already trained weights 
        fine_tune (bool): indicates if we want to make fine tunning
        model_name (str): name of the file where the trained model weights were saved (e.g. trained-model-1.pth)
    Returns:
        torchvision.models.efficientnet.EfficientNet
    """
    if pretrained == True and model_name is not None:
        # load trained weights
        model_path = os.path.join(model_weights_path, trained_model_name)
        pretrained_weights = torch.load(model_path)
        model = get_model(architecture, model_name,pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes)
        model.load_state_dict(pretrained_weights['model_state_dict'])
    else:
        model = get_model(architecture, model_name,pretrained)
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


def build_model_inference(model_path,
                          architecture='EfficientNet', 
                          model_name='efficientnet_b0',
                          num_classes=20, 
                          pretrained=True):
    """
    build model for inference
    """
    pretrained_weights = torch.load(model_path)  
    model = get_model(architecture, model_name,pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=num_ftrs, out_features=num_classes)
    model.load_state_dict(pretrained_weights['model_state_dict'])
    return model
