
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from conabio_ml_vision.datasets import ImageDataset, ImagePredictionDataset

if __name__ == '__main__':
    __spec__ = None
    files_dir = '../../../data/files/missouri_new_files'
    output_dir = '../../../output/dataset'
    # constantes
    crops_dir = os.path.join(files_dir, 'crop_images')
    crops_ds_file = os.path.join(files_dir, 'crops_ds.csv')
    crop_img_ds = ImageDataset.from_csv(crops_ds_file, 
                                    images_dir=crops_dir,
                                    append_label_to_item_path=True, 
                                clean_cat_names=False)

    #df = crop_img_ds.as_dataframe()
    #plt.tight_layout()
    #countplot = sns.countplot(x=df["label"])
    #plt.xticks(rotation=90)
    #fig = countplot.get_figure()
    #fig.savefig(os.path.join(output_dir,"classes_distribution.png")) 
    
    #train
    #crops_dir = os.path.join(files_dir, 'split_crop_images','train')
    crops_ds_file = os.path.join(files_dir, 'split_crops_ds.csv')
    crop_img_ds = ImageDataset.from_csv(crops_ds_file, 
                                    append_label_to_item_path=True, 
                                clean_cat_names=False)

    df = crop_img_ds.as_dataframe()
    df = df[df['partition']=='train']
    plt.tight_layout()
    countplot = sns.countplot(x=df["label"])
    plt.xticks(rotation=90)
    fig = countplot.get_figure()
    fig.savefig(os.path.join(output_dir,"classes_distributio_train.png")) 
    
    #test
    #crops_dir = os.path.join(files_dir, 'split_crop_images','test')
    crops_ds_file = os.path.join(files_dir, 'split_crops_ds.csv')
    crop_img_ds = ImageDataset.from_csv(crops_ds_file, 
                                    append_label_to_item_path=True, 
                                clean_cat_names=False)

    df = crop_img_ds.as_dataframe()
    df = df[df['partition']=='test']
    plt.tight_layout()
    countplot = sns.countplot(x=df["label"])
    plt.xticks(rotation=90)
    fig = countplot.get_figure()
    fig.savefig(os.path.join(output_dir,"classes_distributio_test.png")) 
    
    #validation
    #crops_dir = os.path.join(files_dir, 'split_crop_images','validation')
    crops_ds_file = os.path.join(files_dir, 'split_crops_ds.csv')
    crop_img_ds = ImageDataset.from_csv(crops_ds_file, 
                                    append_label_to_item_path=True, 
                                clean_cat_names=False)

    df = crop_img_ds.as_dataframe()
    df = df[df['partition']=='validation']
    plt.tight_layout()
    countplot = sns.countplot(x=df["label"])
    plt.xticks(rotation=90)
    fig = countplot.get_figure()
    fig.savefig(os.path.join(output_dir,"classes_distributio_val.png")) 
    

    # ploting
    