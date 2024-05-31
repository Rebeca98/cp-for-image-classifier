import torch
import glob as glob

from model import build_model_inference
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from paths import *


if __name__ == "__main__":
    images_path = glob.glob(TEST_DIR+'/*.jpg')
    # Constants.
    IMAGE_SIZE = 580

    # transformation for test set
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943]
    ])
    train_loader, valid_loader, dataset_classes = get_model_dataloaders(batch_size,TRAIN_DIR,TEST_DIR)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    #prediction(img_path, transformer)
    model = build_model_inference(modelname='efficientnet_b0',
                                  num_classes= len(dataset_classes),
                                  pretrained=True,
                                  model_name="model-batch_size_32-optimizer_adam-lr_0.001.pth")
    model.eval()

    pred_dict = {}
    for i in images_path:
        pred_dict[i[i.rfind('/')+1:]] = prediction(i, transform)