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
import matplotlib.pyplot as plt
import random
import os
from helpers.helpers import dir_path
import json
import pandas as pd
from torchvision import datasets

if __name__ == "__main__":
    datapath = "/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/data/processed/calibration"
    outputpath = "/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/output/conformal-alg"
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    # Fix randomness
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    batch_size = 32

    f = open(
        '/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/model_metadata.json')
    model_info_file = json.load(f)
    image_size = model_info_file['image_size']
    num_classes = model_info_file['num_classes']

    transform = get_calib_transform(image_size)
    calib_dataset = datasets.ImageFolder(datapath, transform)
    num_calib = len(calib_dataset)

    VALID_SPLIT = 0.1  # 10% of data used for validation
    num_val = int(num_calib*VALID_SPLIT)
    calib_data, val_data = torch.utils.data.random_split(
        calib_dataset, [num_calib-num_val, num_val])
    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    cudnn.benchmark = True

    # Get the model
    #model = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
    name = 'model'
    modelpath = os.path.join(
        model_info_file['models'][name]['path'], model_info_file['models'][name]['file'])
    model_arch = model_info_file['models'][name]['architecture']

    model = build_model_for_cp(modelpath, model_arch,
                               num_classes, pretrained=True).to(device)

    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'adaptiveness'
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model

    cmodel = ConformalModel(model, calib_loader,kreg=5,lamda=0.01,
                            alpha=0.1, lamda_criterion='size')

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    top1, top5, coverage, size = validate(val_loader, cmodel, print_bool=True)

    print("Complete!")
    num_images = 8
    explore_data, _ = torch.utils.data.random_split(
        val_data, [num_images, num_val-num_images])

    explore_loader = torch.utils.data.DataLoader(
        explore_data, batch_size=1, shuffle=True, pin_memory=True)

    dict_classes = calib_dataset.class_to_idx
    labeldict = {}
    for name, number in dict_classes.items():
        labeldict[number] = name

    mosaiclist = []
    sets = []
    labels = []
    for i, (img, label) in enumerate(explore_loader):
        unnormalized_img = (img * torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))+torch.Tensor(
            [0.485, 0.456, 0.406]).view(-1, 1, 1)
        output, S = cmodel(img.to(device))
        print(label)
        set = [labeldict[s] for s in S[0]]
        sets = sets + [set]
        #labels = labels + [label[0].item()]
        labels = labels + [labeldict[label[0].item()]]
        mosaiclist = mosaiclist + [unnormalized_img]

    mosaiclist = [mosaiclist[i][0]
                  for i in range(len(mosaiclist))]
    grid = torchvision.utils.make_grid(mosaiclist)
    fig, ax = plt.subplots(
        figsize=(min(num_images, 9)*5, np.floor(num_images/9+1)*5))
    ax.imshow(grid.permute(1, 2, 0), interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outputpath, "explore_images_notune.png"))
    dfs = []
    for i in range(len(mosaiclist)):
        dfs.append(pd.DataFrame.from_dict({"Image": [i],
                                           "real-label": [labels[i]],
                                           "predictive-set": [sets[i]],
                                           }))
        print(
            f"Image {i} has label \'{labels[i]}\', and the predictive set is {sets[i]}.")

    df = pd.concat(dfs)
    df.to_csv(os.path.join(outputpath, "explore-results_notune.csv"))
