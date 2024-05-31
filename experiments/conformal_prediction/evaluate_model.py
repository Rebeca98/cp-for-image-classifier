import os
import json
import argparse, configparser
# torch
import torch.backends.cudnn as cudnn
# conformal prediction
from utils_experiments import evaluation_table
from paths import OUTPUT_CP_DIR,MODELS_INFO


if __name__ == "__main__":    
    cudnn.benchmark = True
    
    # config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help='Config file')
    args = parser.parse_args()
    config_file_name = args.config_file.split("/")[-1].split(".")[0]
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("evaluation")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

    # experiment arguments
    image_size = int(args.image_size)
    seed = int(args.random_seed)
    num_classes = int(args.num_classes)
    alphas = eval(args.alphas)
    predictors = eval(args.predictors)
    kregs = eval(args.kregs)
    num_trials = eval(args.num_trials)
    lamdas = eval(args.lamdas)
    randomized = eval(args.randomized)
    pct_paramtune = float(args.pct_paramtune)
    lamda_criterion = eval(args.lamda_criterion)
    bsz = int(args.batch_size)
    strata = eval(args.strata)
    modelnames = eval(args.modelnames)
    data_path_calibration = eval(args.data_path_calibration)
    data_path_test = eval(args.data_path_test)
    args = parser.parse_args()

    
    #output_experiment_csv = output_experiment_path.format(config_file_name)
    output_eval_experiment_pkl = os.path.join(OUTPUT_CP_DIR,config_file_name,'evaluation_table_df.pkl') 
    os.makedirs(os.path.dirname(output_eval_experiment_pkl), exist_ok=True)
    # JSON file that contains directories or paths to trained models.
    
    f = open(MODELS_INFO)
    models_info_dict = json.load(f)
    f.close()
    allow_zero_sets = True
    # Perform the experiment
    evaluation_table_df = evaluation_table(modelnames = modelnames, 
                                           model_info=models_info_dict['models'],
                                           predictors = predictors, 
                                           num_classes = num_classes,
                                           image_size=image_size, 
                                           data_path_calibration = data_path_calibration,
                                           data_path_test = data_path_test, 
                                           bsz = bsz, 
                                           alphas = alphas, 
                                           kregs = kregs,
                                           lamdas = lamdas,
                                           num_trials = num_trials, 
                                           pct_paramtune = pct_paramtune, 
                                           lamda_criterions = lamda_criterion, 
                                           strata = strata)
    
    # save results in csv file
    evaluation_table_df.to_pickle(output_eval_experiment_pkl)
    

