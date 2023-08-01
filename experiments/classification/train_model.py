import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from paths import TRAIN_DIR,VAL_DIR,OUTPUT_DIR,MODEL_PATH
import argparse, configparser
import pandas as pd
from pathlib import Path
#import os


from classification_model.model.train import train
from classification_model.model.train import validate
from classification_model.model.train import EarlyStopper
from classification_model.model.model import build_model
from classification_model.model.datasets import get_model_dataloaders
from classification_model.model.utils import save_model
from classification_model.model.utils import save_plots

if __name__ == '__main__':
    
    # Computation device
    # mps
    #device = ('mps' if torch.backends.mps.is_available() & torch.backends.mps.is_built() else 'cpu') 
    #nvidia
    device = ('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Computation device: {device}")
    
    # get the start time
    st = time.time()

    # model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help='Config file')

    args = parser.parse_args()
    file_path = Path(args.config_file)
    config_file_name = file_path.name.split(".")[0] #args.config_file.split("/")[1].split(".")[0]
    
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("train")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

    batch_size =int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    optimizer_name = str(args.optimizer)
    fine_tune = eval(args.fine_tune)
    patience = int(args.patience)
    image_size = int(args.image_size)
    seed = int(args.random_seed)
    model_name = str(args.model_name)
    architecture = str(args.architecture)
    
    torch.manual_seed(seed)
    # Load the training and validation dataloaders.
    train_loader, valid_loader, dataset_classes = get_model_dataloaders(batch_size,TRAIN_DIR,VAL_DIR,image_size)
    print(f"[INFO]: Class names: {dataset_classes}\n")

    # Build model
    model = build_model(model_weights_path=MODEL_PATH, 
                        architecture=architecture, 
                        model_name=model_name,
                        num_classes=len(dataset_classes), 
                        pretrained=False, # False: train for first time
                        fine_tune=fine_tune).to(device)
    

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    
    # Optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    
    #model_properties = {
    #"batch_size":batch_size,
    #"epochs": epochs,
    #"optimizer":optimizer_name,
    #"lr":lr,
    #"fine_tune":str(fine_tune)    
    #}
    
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Start the training.
    early_stopper = EarlyStopper(patience=patience, delta=10)
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion,device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        #print(
        #    f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        #print(
        #    f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        #print('-'*50)
        if early_stopper.early_stop(valid_epoch_loss):
            print(f"[INFO]: EarlyStopping")             
            break
        time.sleep(5)
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    
    # Save the trained model weights.
    save_model(model, epochs, optimizer, criterion,OUTPUT_DIR,config_file_name)
    
    # Save the loss and accuracy plots.

    save_plots(train_acc, valid_acc, train_loss, valid_loss,OUTPUT_DIR,config_file_name)
    
    if early_stopper.early_stop(valid_epoch_loss):
        save_plots(train_acc, valid_acc, train_loss, valid_loss,OUTPUT_DIR,config_file_name,valid_epoch_acc)


    # valid_epoch_loss print el early stopping
    
    print('[INFO]: TRAINING COMPLETE')