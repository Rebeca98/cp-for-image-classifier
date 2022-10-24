import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as models
from paths import *
matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"{OUTPUT_DIR}/model.pth")
#torch.save(trained_model.state_dict(), 'trained.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/loss.png")


def get_model(modelname='efficientnet_b0', pretrained=True):
    if modelname == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b3':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b4':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b5':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b6':
        model = models.efficientnet_b2(pretrained=pretrained, progress=True)

    elif modelname == 'efficientnet_b7':
        model = models.efficientnet_b3(pretrained=pretrained, progress=True)

    else:
        raise NotImplementedError

    return model
