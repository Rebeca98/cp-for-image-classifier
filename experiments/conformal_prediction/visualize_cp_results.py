import numpy as np
import os
import sys
import json
import pandas as pd
import argparse, configparser
import random
# torch and torchvision
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import datasets
# conformal_classification
from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate
from conformal_classification.utils_experiments import get_calib_transform, build_model_for_cp
# plot libraries
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # paths
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    output_path = os.path.join("..", "results","conformal_prediction")
    models_info = os.path.join("..","files","models_metadata.json")
    
    # device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    
    # enable an automatic algorithm selection process within cuDNN
    cudnn.benchmark = True
    
    # config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help='Config file')
    args = parser.parse_args()
    config_file_name = args.config_file.split("/")[1].split(".")[0]
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("paramater_optimization")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

    # experiment arguments
    image_size = int(args.image_size)
    seed = int(args.random_seed)
    model_name = str(args.model_name)
    num_classes = int(args.num_classes)
    alphas = eval(args.alphas)
    predictors = eval(args.predictors)
    kregs = eval(args.kregs)
    num_trials = int(args.num_trials)
    lamdas = eval(args.lamdas)
    randomized = eval(args.randomized)
    total_conf = int(args.total_conf)
    pct_cal = float(args.pct_cal)
    pct_val = float(args.pct_val)
    pct_paramtune = float(args.pct_paramtune)
    num_classes = int(args.num_classes)
    lamda_criterion = str(args.lamda_criterion)
    bsz = int(args.batch_size)
    strata = eval(args.strata)
    allow_zero_sets = eval(args.allow_zero_sets)
    alpha =  float(args.alpha)
    lamda = float(args.lamda)
    calibration_dataset_path = str(args.calibration_dataset_path)
    args = parser.parse_args()

    # Fix randomness
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # JSON file that contains directories or paths to trained models.
    f = open(models_info)
    models_info_dict = json.load(f)
    f.close()
    modelnames = models_info_dict['models'].keys()
    models_information = [(models_info_dict['models'][name]['file'],
                        models_info_dict['models'][name]['architecture']) for name in modelnames]
    for modelinfo in models_information:
        model = build_model_for_cp(modelinfo[0], architecture=modelinfo[1],
                               num_classes=num_classes, pretrained=True).to(device)
        #model = torch.nn.DataParallel(model)
        model.eval()

        # Conformalize model
        transform = get_calib_transform(image_size)
        calib_dataset = datasets.ImageFolder(calibration_dataset_path, transform)
        num_calib = len(calib_dataset)
        VALID_SPLIT = 0.1  # 10% of data used for validation
        num_val = int(num_calib*VALID_SPLIT)
        calib_data, val_data = torch.utils.data.random_split(calib_dataset, [num_calib-num_val, num_val])
        
        # Initialize loaders
        calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=bsz, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=bsz, shuffle=True, pin_memory=True)
        # Conformalize model
        cmodel = ConformalModel(model, calib_loader, alpha=alpha, lamda=lamda,
                            randomized=randomized, allow_zero_sets=allow_zero_sets)

        print("Model calibrated and conformalized! Now evaluate over remaining data.")
        top1, top5, coverage, size = validate(val_loader, cmodel, print_bool=True)

        print("Complete!")
        
        # visualize results
        num_images = 8
        explore_data, _ = torch.utils.data.random_split(val_data, [num_images, num_val-num_images])

        explore_loader = torch.utils.data.DataLoader(explore_data, batch_size=1, shuffle=True, pin_memory=True)

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

        mosaiclist = [mosaiclist[i][0] for i in range(len(mosaiclist))]
        grid = torchvision.utils.make_grid(mosaiclist)
        fig, ax = plt.subplots(
            figsize=(min(num_images, 9)*5, np.floor(num_images/9+1)*5))
        ax.imshow(grid.permute(1, 2, 0), interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,f"{modelinfo[0]}" ,"explore_images_notune.png"))
        
        # generate a table with results
        dfs = []
        for i in range(len(mosaiclist)):
            dfs.append(pd.DataFrame.from_dict({"Image": [i],
                                            "real-label": [labels[i]],
                                            "predictive-set": [sets[i]],
                                            }))
            print(
                f"Image {i} has label \'{labels[i]}\', and the predictive set is {sets[i]}.")

        df = pd.concat(dfs)
        df.to_csv(os.path.join(output_path,f"{modelinfo[0]}", "explore-results_notune.csv"))
