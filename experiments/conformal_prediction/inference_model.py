import os
import json
import argparse, configparser
# torch
import torchvision
import torch.backends.cudnn as cudnn
# conformal prediction
from utils_experiments import evaluation_table
from paths import OUTPUT_CP_DIR,MODELS_INFO
import torch
import random
import numpy as np
from conformal_classification.evaluate import cp_inference
from params import *
if __name__ == "__main__":    
    cudnn.benchmark = True
    device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')

    # config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
                            "-c", 
                            "--predictor", 
                            type=str, 
                            help='CP methodology', 
                            required=True,
                            choices=['APS', 'RAPS', 'Naive']
                        )
    parser.add_argument("-c", "--num_classes", type=int, help='Number of classes', required=True)
    parser.add_argument("-c", "--alpha", type=float, help='Significance level', required=True)
    parser.add_argument("-c", "--lamda_criterion", type=str,default='size', help='lambda criterion for RAPS')
    parser.add_argument("-c", "--pct_paramtune", type=float,default=0.2, help='Percentage of calibration data for hyperparameter tunning')
    parser.add_argument("-c", "--strata", type=list, help='Strata')
    parser.add_argument("-c", "--randomized", type=list, help='randomization')
    parser.add_argument("-c", "--allow_zero_sets", type=list, help='allow prediction sets of zero size')

    args = parser.parse_args() # Overwrite arguments

    # experiment arguments
    
    seed = 1998
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    image_size = 400
    bsz = 10
    model_evaluation = True
    df_result,conformalized_model = cp_inference(trained_model, 
                 num_classes = args.num_classes,
                 alpha = args.alpha, 
                 image_size= image_size, 
                 predictor = args.predictor,
                data_path_calibration = data_path_calibration, 
                bsz = bsz,
                randomized = args.randomized,
                allow_zero_sets = args.allow_zero_sets,
                pct_paramtune = args.pct_paramtune, 
                lamda_criterion = args.lamda_criterion, 
                strata = args.strata, 
                model_evaluation = model_evaluation,
                data_path_test =data_path_test)

    
    # encapsular esto en una funcion
    # inference data
    transform = get_calib_transform(image_size)
    # calib_dataset = datasets.ImageFolder(CAL_DIR, transform)
    dataset = torchvision.datasets.ImageFolder(inference_data, transform)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True)
    

    
    dict_classes = dataset_loader.class_to_idx
    labeldict ={}
    for name, number in dict_classes.items():
        labeldict[number] = name

    conformalized_model.eval()
        
    prediction_sets = []    
    for i, (x, target) in enumerate(dataset_loader):
        target = target.to(device)
        # compute output
        # S: prediction set
        # compute plat scaling and conformal pred. algorithm
        output, S = conformalized_model(x.to(device))

