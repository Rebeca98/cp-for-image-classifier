import argparse
import sys

from conabio_ml_vision.datasets import ImageDataset
from params import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_perc", default=0.6,type=float)
    parser.add_argument("--test_perc", default=0.3, type=float) #calibration dataset
    parser.add_argument("--val_perc", default=0.1, type=float)
    parser.add_argument("--random_state", default=42, type=float)
    args = parser.parse_args()
    
    crops_ds = ImageDataset.from_csv(lila_crops_ds_csv,images_dir = lila_crops_dir)

    crops_split_ds = crops_ds.split(train_perc=args.train_perc,
                    test_perc=args.test_perc,
                    val_perc = args.val_perc,
                    group_by_field='seq_id',
                    stratify=True,
                   random_state=args.random_state)
    crops_split_ds.to_csv(crops_split_ds_csv)
    crops_split_ds.to_folder(lila_crops_split_dir)