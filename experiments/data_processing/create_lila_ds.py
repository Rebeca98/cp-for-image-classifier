import argparse
import os

from conabio_ml_vision.datasets import LILADataset
from conabio_ml_vision.utils import lila_utils
from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns

from params import *


from conabio_ml_vision.datasets import ImagePredictionDataset
from conabio_ml.utils.utils import is_array_like


def get_detections_ds_for_collection(collection,
                                     md_result_jsons,
                                     threshold_mdv5,
                                     threshold_mdv4,
                                     detection_cats='animal'):
    dets_filename = md_result_jsons[collection]
    if not is_array_like(dets_filename):
        dets_ds = ImagePredictionDataset.from_json(
            dets_filename, min_score=threshold_mdv5, validate_filenames=False)
    else:
        dss = []
        for dets_fname in dets_filename:
            temp_det_ds = ImagePredictionDataset.from_json(
                dets_fname, min_score=threshold_mdv4, validate_filenames=False)
            dss.append(temp_det_ds)
        dets_ds = ImagePredictionDataset.from_datasets(*dss)
    dets_ds.filter_by_categories(detection_cats, mode='include', inplace=True)
    return dets_ds

def create_complete_lila_ds(datasets_names=None,
                            threshold_mdv5=0.4,
                            threshold_mdv4=0.8,
                            force=False):
    img_level_results_dir = os.path.dirname(lila_imgs_csv)
    bbox_level_results_dir = os.path.dirname(lila_bboxes_csv)

    if datasets_names is not None:
        _datasets_names = [x.strip() for x in datasets_names.split(',')]
        all_json_files_dict = lila_utils.get_all_json_files(
            metadata_dir, collections=_datasets_names, delete_zip_files=False)
        _datasets_names_not_bbox = [x for x in _datasets_names if not x.endswith('_bbox')]
        md_result_jsons = lila_utils.get_all_md_result_jsons(
            md_results_paths, metadata_dir, collections=_datasets_names_not_bbox,
            delete_zip_files=False)
    else:
        all_json_files_dict = lila_utils.get_all_json_files(metadata_dir, delete_zip_files=False)
        md_result_jsons = lila_utils.get_all_md_result_jsons(
            md_results_paths, metadata_dir, delete_zip_files=False)
        _datasets_names = list(all_json_files_dict.keys())

    all_img_level_dss = {}
    all_obj_level_dss = {}
    for collection in _datasets_names:
        col_fname = collection.replace(' ', '_')
        img_level_coll_csv = os.path.join(img_level_results_dir, f"{col_fname}.csv")
        obj_level_coll_csv = os.path.join(bbox_level_results_dir, f"{col_fname}.csv")

        if not force and os.path.isfile(img_level_coll_csv) and os.path.isfile(obj_level_coll_csv):
            img_level_ds = LILADataset.from_csv(img_level_coll_csv)
            obj_level_ds = LILADataset.from_csv(obj_level_coll_csv)
        else:
            json_filename = all_json_files_dict[collection]
            ds = LILADataset.from_json(
                source_path=json_filename,
                exclude_categories=['homo sapiens'],
                mapping_classes_csv=mapping_classes_csv,
                collection=collection,
                exclude_invalid_scientific_names=True,
                validate_filenames=False)

            if collection.endswith('_bbox'):
                # if 'caltech' not in collection.lower():
                # TODO: to include caltech_bbox collection it is necessary remove duplicates with caltech
                # There are only ~700 useful annotations in caltech_bbox not present in caltech
                continue
                # obj_level_ds = ds
            else:
                if ds.is_detection_dataset():
                    ds_w_bboxes = ds.filter_by_column('bbox', '', mode='exclude', inplace=False)
                    ds_wo_bboxes = ds.filter_by_column('bbox', '', mode='include', inplace=False)
                else:
                    ds_wo_bboxes = ds

                if ds_wo_bboxes.get_num_rows() > 0:
                    dets_ds = get_detections_ds_for_collection(
                        collection, md_result_jsons, threshold_mdv5, threshold_mdv4)
                    imgs_ids_wo_bboxes = ds_wo_bboxes.get_column_values('image_id')
                    dets_wo_bboxes_ds = dets_ds.filter_by_column(
                        'image_id', imgs_ids_wo_bboxes, mode='include', inplace=False)

                    min_score_dets = (
                        threshold_mdv4 if 'serengeti' in collection.lower() else threshold_mdv5)
                    ds_from_dets_wo_bboxes = create_obj_level_ds_from_dets_and_anns(
                        detections=dets_wo_bboxes_ds,
                        annotations=ds_wo_bboxes,
                        det_categories=['animal'],
                        min_score_detections=min_score_dets)

                    if ds.is_detection_dataset():
                        obj_level_ds = LILADataset.from_datasets(
                            ds_w_bboxes, ds_from_dets_wo_bboxes)
                    else:
                        obj_level_ds = ds_from_dets_wo_bboxes
                else:
                    obj_level_ds = ds

            obj_level_ds.to_csv(obj_level_coll_csv)
            img_level_ds = obj_level_ds.create_image_level_ds()
            img_level_ds.to_csv(img_level_coll_csv)

        all_img_level_dss[collection] = img_level_ds
        all_obj_level_dss[collection] = obj_level_ds

    if datasets_names is None:
        all_img_level_ds_list = list(all_img_level_dss.values())
        final_img_level_ds = LILADataset.from_datasets(*all_img_level_ds_list)
        final_img_level_ds.set_images_info_field_by_expr(
            'location',
            lambda row: f"{row['collection'].lower().replace(' ', '')}-{row['location']}",
            inplace=True)
        final_img_level_ds.to_csv(lila_imgs_csv)

        all_obj_level_ds_list = list(all_obj_level_dss.values())
        final_obj_level_ds = LILADataset.from_datasets(*all_obj_level_ds_list)
        final_obj_level_ds.set_images_info_field_by_expr(
            'location',
            lambda row: f"{row['collection'].lower().replace(' ', '')}-{row['location']}",
            inplace=True)
        final_obj_level_ds.to_csv(lila_bboxes_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--create_complete_lila_ds", default=False, action="store_true")

    parser.add_argument(
        '--datasets_names', default=None, type=str,
        help=(f'Indicates the collections to be analyzed by the current process, separated by '
              f'commas. Any combination of the following can be selected: '
              f'NACTI,NACTI_bbox,WCS Camera Traps,WCS Camera Traps_bbox,'
              f'Wellington Camera Traps,Island Conservation Camera Traps,'
              f'Idaho Camera Traps,Caltech Camera Traps,Caltech Camera Traps_bbox,ENA24,'
              f'Missouri Camera Traps,Snapshot Serengeti,Snapshot Serengeti_bbox,',
              f'Snapshot Karoo,Snapshot Kgalagadi,Snapshot Enonkishu,Snapshot Camdeboo,'
              f'Snapshot Mountain Zebra,Snapshot Kruger,SWG Camera Traps,'
              f'SWG Camera Traps_bbox,Orinoquia Camera Traps,Channel Islands Camera Traps')
    )

    args = parser.parse_args()

    if args.create_complete_lila_ds:
        create_complete_lila_ds(datasets_names=args.datasets_names)
