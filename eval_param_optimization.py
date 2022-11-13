import argparse
import pandas as pd
import random

# torch
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
from helpers.helpers import dir_path
#import experiments
from conformal_classification.experiments import create_df_evaluation

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

    # experiment output
    experiment_name = "optim-parameters-exp1"
    path = os.path.join('output/conformal-alg', experiment_name) + ".csv"

    alpha_table = 0.1
    try:
        df = pd.read_csv(path)
    except:
        # Configure experiment
        f = open(
            '/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/model_metadata.json')
        model_info_file = json.load(f)
        image_size = model_info_file['image_size']
        num_classes = model_info_file['num_classes']
        alphas = [0.05, 0.10]
        predictors = ['Fixed', 'Naive', 'APS', 'RAPS']

        num_trials = 10
        kreg = [None]
        lamda = [None]
        randomized = True
        total_conf = 1450
        pct_cal = 0.6
        pct_val = 0.4
        pct_paramtune = 0.33
        # est deberia estar especificado en el json
        num_classes = 14

        bsz = 32
        cudnn.benchmark = True

        # Perform the experiment
        df = create_df_evaluation(model_info_file, args.data, num_classes, alphas, predictors, kreg, lamda,
                                  randomized, total_conf, pct_cal, pct_val, bsz,
                                  image_size, num_trials, pct_paramtune, pretrained=True)
        df.to_csv(path)
