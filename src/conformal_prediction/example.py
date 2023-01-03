import random
import os
import sys
import argparse
# torch and torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from datasets import get_cmodel_data_loaders, get_cmodel_dataset

import numpy as np
# conformal_classification
sys.path.append('/cp-for-image-classifier/conformal_classification')
from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate
from conformal_classification.utils_experiments import get_calib_transform, build_model_for_cp
from paths import *


num_classes = 20
trained_model_name = 'model-batch_size_32-optimizer_adam-lr_0.001.pth'

parser = argparse.ArgumentParser(
    description='Conformalize Torchvision Model on Imagenet')
# parser.add_argument('data', metavar='CALIBDIR',
#                    help='path to calibration data')

# parser.add_argument('model', metavar='MODELPATH',
#                    help='path from model we will use for inference')
parser.add_argument('--batch_size', metavar='BSZ',
                    help='batch size', default=20)
parser.add_argument('--num_workers', metavar='NW',
                    help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB',
                    help='number of calibration points', default=1450)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)


if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    args = parser.parse_args()
    # Fix randomness
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
    transform = get_calib_transform(580)

    # Get the conformal calibration dataset
    #VALID_SPLIT = 0.1  # 10% of data used for validation
    #num_val = int(args.num_calib*VALID_SPLIT)

    #calib_data, val_data = torch.utils.data.random_split(
    #    torchvision.datasets.ImageFolder(CALIB_PATH, transform), [args.num_calib-num_val, num_val])

    # Initialize loaders
    #calib_loader = torch.utils.data.DataLoader(
    #    calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(
    #    val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    calib_data, calib_val_data = get_cmodel_dataset()

    calib_loader , val_loader = get_cmodel_data_loaders(calib_data, calib_val_data)
    cudnn.benchmark = True

    # Get the model
    #model = torchvision.models.resnet152(pretrained=True, progress=True).cuda()

    model = build_model_for_cp(os.path.join(MODEL_PATH, trained_model_name), architecture='efficientnet_b0',
                               num_classes=num_classes, pretrained=True).to(device)
    #model = torch.nn.DataParallel(model)
    model.eval()

    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0,
                           randomized=randomized, allow_zero_sets=allow_zero_sets)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validate(val_loader, model, print_bool=True)

    print("Complete!")
