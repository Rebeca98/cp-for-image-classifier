import os
from pathlib import Path

from conabio_ml_vision.utils import coords_utils
from conabio_ml_vision.datasets import ImageDataset, ImagePredictionDataset
from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns


if __name__ == '__main__':
    __spec__ = None
    images_dir = '../../../data/raw'
    files_dir = '../../../data/files/missouri_new_files'
    os.makedirs(files_dir, exist_ok=True)
    processed_dir = '../../../data/processed'
    crops_dir = os.path.join(processed_dir, 'crop_images')
    os.makedirs(crops_dir, exist_ok=True)
    csv_path = os.path.join(files_dir, 'imgs_ds.csv')
    crops_ds_file = os.path.join(files_dir, 'crops_ds.csv')
    imgs_ds_file = os.path.join(files_dir, 'images_ds.csv')
    dets_ds_file = os.path.join(files_dir, 'dets_ds.csv')


    animal_dets_obj_level_ds = create_obj_level_ds_from_dets_and_anns(detections=dets_ds_file,
                                           annotations=imgs_ds_file,
                                           images_dir=images_dir,
                                           inherit_fields=[],
                                           inherit_partitions=False,
                                           annot_categories=None,
                                           exclude_annot_categories=None,
                                           det_categories=None,
                                           exclude_det_categories=None,
                                           min_score_detections=0.001,
                                           include_detection_score=True)

    crops_ds = animal_dets_obj_level_ds.create_classif_ds_from_bboxes_crops(
    dest_path=None, allow_label_empty=True) # esto ayuda cuando tenemos los resultados del megadetector y mis etiquetas del dataset
    crops_ds.to_csv(crops_ds_file)
    crops_ds.to_folder(crops_dir)

    