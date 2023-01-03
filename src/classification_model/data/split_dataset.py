import os
from pathlib import Path

from conabio_ml_vision.datasets import ImageDataset, ImagePredictionDataset

if __name__ == '__main__':
    __spec__ = None
    files_dir = '../../../data/processed'
    # constantes
    crops_dir = os.path.join(files_dir, 'crop_images')
    crops_ds_file = os.path.join(files_dir, 'crops_ds.csv')
    split_crops_csv = os.path.join(files_dir,'split_crops_ds.csv')
    split_crops_folder = os.path.join(files_dir, 'split_crop_images')
    os.makedirs(split_crops_folder, exist_ok=True)
    crop_img_ds = ImageDataset.from_csv(crops_ds_file, 
                                    images_dir=crops_dir,
                                    append_label_to_item_path=True, 
                                clean_cat_names=False)

    split_img_ds = crop_img_ds.split(train_perc = 0.5,
                                    test_perc = 0.2,
                                    val_perc  = 0.3,#calibration set
                                    stratify=True, 
                                    group_by_field='seq_id')

    split_img_ds.to_csv(split_crops_csv,remove_label_to_item_path=True)
    split_img_ds.to_folder(split_crops_folder)