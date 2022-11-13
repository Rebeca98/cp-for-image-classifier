import torch
import numpy as np
import glob as glob
import os
from PIL import Image

from model import build_model_inference
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from paths import *

# Class names.
class_names = ['agouti', 'collared_peccary', 'common_opossum', 'european_hare', 'ocelot', 'paca', 'red_brocket_deer',
               'red_fox', 'red_squirrel', 'roe_deer', 'spiny_rat', 'white-nosed_coati', 'white_tailed_deer', 'wild_boar']

# Load the trained model.
# from torchvision import models model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#print('Loading trained model weights...')
# model.load_state_dict(checkpoint['model_state_dict'])


def prediction(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
    input = Variable(image_tensor)
    with torch.no_grad():
        output = model(input)
    index = output.data.numpy().argmax()

    pred = class_names[index]
    return pred


def prediction(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
    input = Variable(image_tensor)
    with torch.no_grad():
        output = model(input)
    index = output.data.numpy().argmax()

    pred = class_names[index]
    return pred


# Iterate over all the images and do forward pass.
# for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    #gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0]
    # Read the image and create a copy.
    #image = cv2.imread(image_path)
    #image = Image.open(image_path)
    #orig_image = image.copy()

    # Preprocess the image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225]
    #    )
    # ])
    #image = transform(image)
    #image = torch.unsqueeze(image, 0)
    #image = image.to(DEVICE)

    # Forward pass throught the image.
    # with torch.no_grad():
    #    outputs = model(image)
    #outputs = outputs.detach().numpy()
    #pred_class_name = class_names[np.argmax(outputs[0])]
    # print(outputs)
    #pred_class_name = np.argmax(outputs[0])
    #print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    #print(f" Pred: {pred_class_name.lower()}")
    # Annotate the image with ground truth.
    # cv2.putText(
    #    orig_image, f"GT: {gt_class_name}",
    #    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #    1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
    # )
    # Annotate the image with prediction.
    # cv2.putText(
    #    orig_image, f"Pred: {pred_class_name.lower()}",
    #    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
    #    1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
    # )
    #cv2.imwrite(f"../outputs/{gt_class_name}.png", orig_image)

if __name__ == "__main__":
    images_path = glob.glob(TEST_PATH_J+'/*.jpg')
    # Constants.
    IMAGE_SIZE = 580

    # transformation for test set
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.3768, 0.3809, 0.3522],[0.1951,0.1968,0.1943]
    ])
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    #prediction(img_path, transformer)
    model = build_model_inference(
        modelname='efficientnet_b0', num_classes=14, pretrained=True).to(device)  # 'weights'
    model.eval()

    pred_dict = {}
    for i in images_path:
        pred_dict[i[i.rfind('/')+1:]] = prediction(i, transform)
  