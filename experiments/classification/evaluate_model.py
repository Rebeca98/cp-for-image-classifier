import torch
from classification_model.model.datasets import get_model_dataloaders_evaluation
from classification_model.model.model import build_model_inference
from classification_model.model.evaluation import get_evaluate_metrics

import argparse, configparser
import pandas as pd
import os
import json

from paths import TEST_DIR,OUTPUT_DIR,MODELS_INFO

if __name__=="__main__":
  # results path
  f = open(MODELS_INFO)
  models_info_dict = json.load(f)
  f.close()

  
  # Load the trained model.
  device = ('cuda' if torch.cuda.is_available() else 'cpu')

  # model parameters
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config_file", type=str, help='Config file')

  args = parser.parse_args()
  config_file_name = args.config_file.split("/")[1].split(".")[0]
  config = configparser.ConfigParser()
  config.read(args.config_file)
  defaults = {}
  defaults.update(dict(config.items("Evaluation")))
  parser.set_defaults(**defaults)
  args = parser.parse_args() # Overwrite arguments

  batch_size =int(args.batch_size)
  image_size = int(args.image_size)
  seed = int(args.random_seed)
  model_name = str(args.model_name)

  # paths
  model_path = models_info_dict["models"][config_file_name]["file_path"]
  evaluation_dir_results = os.path.join(OUTPUT_DIR,config_file_name,"evaluation")
  os.makedirs(evaluation_dir_results, exist_ok=True)
  torch.manual_seed(seed)
  # Dataloader
  test_loader, dataset_classes = get_model_dataloaders_evaluation(batch_size,TEST_DIR,image_size,subset=True)
  # build model
  model = build_model_inference(model_path,
                          architecture='EfficientNet', 
                          model_name ='efficientnet_b0',
                          num_classes =len(dataset_classes), 
                          pretrained =True)

  n_sample = 1000
  df_eval,cf_matrix = get_evaluate_metrics(model, test_loader, dataset_classes=dataset_classes, device=device,n_sample = n_sample)
 
  df_eval.to_csv(os.path.join(evaluation_dir_results,f"metrics.csv"))  
  cf_matrix.savefig(os.path.join(evaluation_dir_results,f"confusion_matrix.png"))
