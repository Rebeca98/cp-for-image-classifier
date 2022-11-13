import argparse
# torch and torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import numpy as np
# conformal_classification
from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate
from conformal_classification.utils_experiments import get_calib_transform, build_model_for_cp

import random
import os

from helpers.helpers import dir_path

parser = argparse.ArgumentParser(
    description='Conformalize Torchvision Model on Imagenet')
# parser.add_argument('data', metavar='CALIBDIR',
#                    help='path to calibration data')
parser.add_argument(
    '-data', '--data', type=dir_path,
    dest='data',
    help='path to calibration data'
)
parser.add_argument(
    '-model', '--model', type=dir_path,
    dest='model',
    help='path to model'
)

parser.add_argument(
    '-modelname', '--modelname', type=str, dest='modelname',
    help='name of the model', default='model.pth'
)

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
    transform = get_calib_transform()

    # Get the conformal calibration dataset
    VALID_SPLIT = 0.1  # 10% of data used for validation
    num_val = int(args.num_calib*VALID_SPLIT)

    calib_data, val_data = torch.utils.data.random_split(
        torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib-num_val, num_val])

    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    cudnn.benchmark = True

    # Get the model
    #model = torchvision.models.resnet152(pretrained=True, progress=True).cuda()

    model = build_model_for_cp(os.path.join(args.model, args.modelname), modelname='efficientnet_b0',
                               num_classes=14, pretrained=True).to(device)
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
