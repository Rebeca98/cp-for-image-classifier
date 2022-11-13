import numpy as np
import random
import argparse
import json
# torch and torchvision
import torch

from conformal_classification.experiments import create_df_sizes_topk

from helpers.helpers import dir_path

#from plots_and_tables import plot_empirical_cov_sz
import os
import pandas as pd


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
    '-modelpath', '--modelpath', type=dir_path,
    dest='modelpath',
    help='path to model'
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
    experiment_name = "topk_size_exp1"
    path = os.path.join('output/conformal-alg', experiment_name) + ".pkl"

    # Configure experiment
    f = open(
        '/Users/rebecaangulorojas/Desktop/TESIS/cp-for-image-classifier/model_metadata.json')
    model_info_file = json.load(f)
    image_size = model_info_file['image_size']
    num_classes = model_info_file['num_classes']
    total_conf = 1405
    pct_cal = 0.4
    pct_val = 0.4
    alphas = [0.1]
    predictors = ['Naive', 'APS', 'RAPS']
    lambdas = [0.01, 0.1, 1]
    kreg = [5]
    randomized = True
    bsz = 16
    df = create_df_sizes_topk(model_info_file, args.data, num_classes, alphas, predictors, lambdas, kreg,
                              randomized, total_conf, pct_cal, pct_val, bsz,
                              image_size, pretrained=True)
    f.close()
    df.to_pickle(path, compression='infer', protocol=5, storage_options=None)
