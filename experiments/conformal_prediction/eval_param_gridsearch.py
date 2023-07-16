import random
import os
import json
import numpy as np
import argparse, configparser
# torch
import torch
import torch.backends.cudnn as cudnn
# conformal prediction
from conformal_classification.experiments import create_evaluation_table


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

    # experiment output
    experiment_name = f"gridsearch-parameters-exp1-{config_file_name}"
    output_experiment_path = os.output_experiment_path.join('output/conformal-alg', experiment_name) + ".csv"
    
    # JSON file that contains directories or paths to trained models.
    f = open(models_info)
    models_info_dict = json.load(f)
    f.close()

    # Perform the experiment
    df = create_evaluation_table(models_info_dict, 
                                 calibration_dataset_path, num_classes, alphas, predictors, kregs, lamdas,
                                  randomized, total_conf, pct_cal, pct_val, bsz,
                                  image_size, num_trials, pct_paramtune, pretrained=True)
    # save results in csv file
    df.to_csv(output_experiment_path)
