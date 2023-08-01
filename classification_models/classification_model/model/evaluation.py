import torch
from torchmetrics.classification import Recall, Precision,Accuracy,F1Score,MulticlassConfusionMatrix
from tqdm.auto import tqdm
import pandas as pd

# requierements: !pip install torchmetrics
# debemos de suponer que mis datos de testing vienen de la misma distribucion
IMAGE_SIZE = 580
BATCH_SIZE = 32
NUM_WORKERS = 0

def _get_confusion_matrix(preds,target,num_classes,dataset_classes,device):
    
    """
    dataset_classes: list
    """
    metric = MulticlassConfusionMatrix(num_classes,normalize='true').to(device)
    metric.update(preds, target)  
    #fig_, ax_ = metric.plot()
    #dataframe = pd.DataFrame(cf_matrix, index=dataset_classes, columns=dataset_classes)
    fig, ax = metric.plot()
    # Create a figure
    #plt.figure(figsize=(8, 6))
    # Create heatmap
    #sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")
    #plt.title("Confusion Matrix") 
    #plt.tight_layout()
    #plt.savefig('my_plot.png')
    return fig

def _get_metrics(preds, target,num_classes,device):
    precision = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
    precision_score = precision(preds, target).item()
    recall = Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)
    recall_score = recall(preds, target).item()
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    accuracy_score = accuracy(preds, target).item()
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)
    f1_score = f1(preds, target).item()
    return precision_score, recall_score, accuracy_score, f1_score

def get_evaluate_metrics(model, testloader, dataset_classes, **kwargs):
    num_classes = len(dataset_classes)
    device = kwargs.get('device', 'cpu')
    preds,target= _evaluate(model, testloader, device)
    precision_score, recall_score, accuracy_score, f1_score = _get_metrics(preds, target,num_classes,device)
    # List of column names
    metrics_dict = {'precision':[precision_score], 
                      'recall':[recall_score],
                      'accuracy':[accuracy_score],
                      'F1-score':[f1_score]}
  # Create an empty DataFrame with only column names
    df = pd.DataFrame.from_dict(metrics_dict)
    cf_matrix = _get_confusion_matrix(preds, target, num_classes, dataset_classes, device)
    return df,cf_matrix

def _evaluate(model, testloader, device):
    #if torch.cuda.is_available():
    #device = kwargs.get('device', 'cpu')
    model = model.to(device)
    model.eval()
    print('Evaluation')
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
    preds = torch.cat(preds_list)
    target = torch.cat(target_list)
    return preds, target