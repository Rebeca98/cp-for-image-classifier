import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as models
matplotlib.style.use('ggplot')
import os

# deprecated
def get_model_name(model_properties: dict):
    current_model_name = "model"
    for key in model_properties.keys():
        current_model_name += "-" + key + "_" + str(model_properties[key])
    return current_model_name


def save_model(model, 
               epochs, 
               optimizer, 
               criterion, 
               training_time,
               dir_path,
               config_file_name):

    """
    Function to save the trained model.
    model: Pytorch model object 
    epochs: int,
    optimizar: pytorch optimizer object
    criterion: torch.nn loss criterion (e.g nn.CrossEntropyLoss())
    dir_path: path to save trained model weights
    config_file_name: str 
        name of the config.ini file with parameter configuration

    """
    directory = os.path.join(dir_path,config_file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'training_time_seconds': training_time  # Save training time
    }, os.path.join(directory,"trained.pth"))
#torch.save(trained_model.state_dict(), 'trained.pth')


def save_plots(train_acc, 
               valid_acc, 
               train_loss, 
               valid_loss,
               output_dir,
               config,
               **kwargs):
    """
    Function to save the loss and accuracy plots.
    """
    valid_epoch_acc = kwargs.get('valid_epoch_acc', None)
    # accuracy plots
    directory = os.path.join(output_dir,config)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
    
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{directory}/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    if valid_epoch_acc is not None:
        plt.axvline(valid_epoch_acc, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f"{directory}/loss.png")


def get_model(architecture,model_name,pretrained):
    """ 
    function that downloads the efficientnet pretrained weights
    architetcture: str (e.g. 'EfficientNet')
    model_name = 'efficientnet_b0'
    pretrained: bool (e.g. True)
    """
    efficientnet_models = ['efficientnet_b0','efficientnet_b1','efficientnet_b2']

    if model_name in efficientnet_models:
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT',
                                        pretrained=pretrained,
                                        progress=True)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT',
            pretrained=pretrained, 
            progress=True)

        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT',
                                            pretrained=pretrained, 
                                        progress=True)
        else:
            raise NotImplementedError
            
    else:
        raise NotImplementedError

    return model
1
