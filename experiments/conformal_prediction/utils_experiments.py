import pandas as pd
from tqdm import tqdm
import itertools
# Torch
import torch
import time
import numpy as np
import os

from conformal_classification.utils import get_calib_transform 
from conformal_classification.evaluate import get_logits_model
from conformal_classification.evaluate import evaluate_trials
from typing import List,Optional
from conformal_classification.evaluate import get_logits_model
from conformal_classification.evaluate import get_logits

#device = ('cuda' if torch.cuda.is_available() else 'cpu')
#device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu')
# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.


# report median size-stratified coverage violation of RAPS and APS
# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.


def evaluation_table(modelnames:list,predictors:list, num_classes:int,image_size:int,  model_info:dict,
                    data_path_calibration:str,data_path_test:str, bsz:int, alphas:List[float], kregs:List[int],lamdas:List[float],
                    num_trials:List[int], pct_paramtune:float, lamda_criterions:List[str], strata:Optional[List[List[int]]] = None):
        #model_path:str, model_name:str,
    randomized = True
    allow_zero_sets = False
    df_evaluate_models = []

    start_time = time.time()  # Inicia el cronómetro
    for modelname in modelnames:
        transform = get_calib_transform(image_size)
        
        cal_logits, model = get_logits_model(model_path = model_info[modelname]['file_path'],
                                    model_name =  model_info[modelname]['architecture'], 
                                    data_path = data_path_calibration, 
                                    transform = transform, 
                                    bsz = bsz, 
                                    num_classes = num_classes)
        test_logits = get_logits(model_path = model_info[modelname]['file_path'],
                                model_name = model_info[modelname]['architecture'],
                                data_path = data_path_test,
                                transform = transform,
                                bsz = bsz,
                                num_classes = num_classes)

        if 'RAPS' in predictors:
            params = list(itertools.product(alphas,kregs,lamdas,predictors,[modelname],num_trials,lamda_criterions))
        else:
            params = list(itertools.product(alphas,kregs,lamdas,predictors,[modelname],num_trials,['size']))
        m = len(params)
        for i in range(m):
            alpha,kreg,lamda,predictor,name,_num_trials,lamda_criterion = params[i]
            print(f'Model: {name} | Desired coverage: {1-alpha}')
            df_evaluate_trial = evaluate_trials(model = model, 
                                                cal_logits = cal_logits, 
                                                test_logits = test_logits, 
                                                bsz= bsz, 
                                                num_trials = _num_trials, 
                                                alpha = alpha, 
                                                kreg = kreg, 
                                                lamda = lamda,
                                                randomized = randomized, 
                                                allow_zero_sets = allow_zero_sets,  
                                                pct_paramtune = pct_paramtune,  
                                                predictor = predictor, 
                                                lamda_criterion = lamda_criterion,
                                                strata = strata, 
                                                num_classes = num_classes)
            df_evaluate_trial['alpha'] = alpha
            df_evaluate_trial['predictor'] = predictor
            df_evaluate_trial['lamda'] = lamda
            df_evaluate_trial['kreg'] = kreg
            df_evaluate_trial['model'] = name

            df_evaluate_models.append(df_evaluate_trial)
    
    df_evaluation_all = pd.concat(df_evaluate_models).reset_index()
    end_time = time.time()  # Detiene el cronómetro
    elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

    print(f'Tiempo total de ejecución: {elapsed_time:.2f} segundos')

    return df_evaluation_all





    
