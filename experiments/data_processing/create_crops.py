import argparse
import sys

from conabio_ml_vision.datasets import LILADataset, ImagePredictionDataset, ImageDataset
from conabio_ml_vision.models import run_megadetector_inference
from conabio_ml_vision.utils.aux_utils import eval_multiclass
from conabio_ml_vision.utils.scripts import get_samples_counts_by_partitions
from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns

from params import *

if __name__ == "__main__":
    __spec__ = None
    parser = argparse.ArgumentParser()
    # region LILA
    #parser.add_argument("--create_lila_dataset", default=False, action="store_true")
    parser.add_argument("--download_lila_imgs", default=False, action="store_true")
    parser.add_argument("--create_lila_dataset_crops", default=False, action="store_true")
    parser.add_argument("--use_azcopy_for_download", default=False, action="store_true")
    # endregion
    
    args = parser.parse_args()

    if args.download_lila_imgs:
        lila_ds_csv = lila_dataset_csv
        lila_ds = LILADataset.from_csv(lila_imgs_csv)


        lila_ds.download(
            dest_path=lila_images_dir,
            num_tasks=None,
            task_num=None,
            azcopy_batch_size=azcopy_batch_size,
            azcopy_exec=azcopy_exec,
            separate_in_dirs_per_collection=True,
            use_azcopy_for_download=args.use_azcopy_for_download)

    if args.create_lila_dataset_crops:
        lila_ds = LILADataset.from_csv(lila_dataset_csv, images_dir=lila_images_dir,not_exist_ok=True) #fix de JC
        crops_ds = lila_ds.create_classif_ds_from_bboxes_crops(
            dest_path=lila_crops_ds_csv,
            force_crops_creation=True,
            not_exist_ok=True)
        crops_ds.to_csv(lila_crops_ds_csv)

    
    

    
