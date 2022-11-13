import argparse
import pandas as pd
import random
import json
import numpy as np
import os
#torch 
import torch.backends.cudnn as cudnn
import torch
#conformal classification library
from conformal_classification.experiments import create_df_violation
#helpers
from helpers.helpers import dir_path

parser = argparse.ArgumentParser(
    description='Conformalize Torchvision Model on Imagenet')
parser.add_argument(
    '-data', '--data', type=dir_path,
    dest='data',
    help='path to calibration data'
)
if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    args = parser.parse_args()
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    #models information
    f = open(
        '/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/model_metadata.json')
    model_info_file = json.load(f)
    image_size = model_info_file['image_size']
    num_classes = model_info_file['num_classes']
    # experiment output
    experiment_name = "violation-exp1"
    path = os.path.join('output/conformal-alg', experiment_name) + ".csv"
    ### Configure experiment
    alphas = [0.05, 0.10]
    randomized = True
    num_trials = 10
    pct_paramtune = 0.33
    num_classes = 14
    total_conf = 1450
    pct_cal = 0.6
    pct_val = 0.4
    bsz = 32
    strata = [[0,1],[2,3],[4,6],[7,10],[11,14]] #according to the number of classes
    cudnn.benchmark = True
    try:
        df = pd.read_csv(path)
    except:
        df = create_df_violation(model_info_file, args.data, num_classes, total_conf, pct_cal, pct_val, alphas, randomized, bsz, image_size, num_trials, strata, pct_paramtune)
        df.to_csv(path)
    f.close()
    