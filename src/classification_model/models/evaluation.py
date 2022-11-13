from torchmetrics.functional import precision_recall
import torch
from torchvision import transforms, datasets
from model import build_model,build_model_inference
from paths import *
from torchvision.transforms.transforms import RandomVerticalFlip
from tqdm.auto import tqdm
from torchmetrics.functional import precision_recall
# requierements: !pip install torchmetrics
# debemos de suponer que mis datos de testing vienen de la misma distribucion
IMAGE_SIZE = 580
BATCH_SIZE = 32
NUM_WORKERS = 0


def evaluate(model, testloader):
    model.eval()
    print('Validation')
    counter = 0
    preds_list = []
    target_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            preds_list.append(preds)
            target_list.append(labels)

    return preds_list, target_list
if __name__=="__main__":
  # load images with from Folder para obtener las clases
  # ver jpnotebook sin titulo aqui hay ejemplos de como se ve ese dataste
  # Load the trained model.
  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  model = build_model_inference(modelname='efficientnet_b0', num_classes=14, pretrained=True).to(device)  # 'weights'
  model.eval()

  transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
  dataset_test = datasets.ImageFolder(
        TEST_PATH, 
        transform=(transform)
    )
  test_loader = torch.utils.data.DataLoader(dataset_test,
                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
  #predictions
  preds_list, target_list = evaluate(model, test_loader)
  # concat all predictions to get metrics
  preds = torch.cat(preds_list)
  target = torch.cat(target_list)
  precision, recall = precision_recall(preds, target, average='macro', num_classes=14)
  print("from a test dataset of size:", len(dataset_test), "\n Precision: ", precision, "\n Recall: ", recall)

  