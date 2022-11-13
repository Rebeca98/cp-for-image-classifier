import os
from pathlib import Path

from conabio_ml_vision.utils import coords_utils
from conabio_ml_vision.datasets import ImageDataset, ImagePredictionDataset
from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns

conabio_ml_path = '/LUSTRE/users/ecoinf_admin/conabio_ml_vision'

json_path = os.path.join(
    conabio_ml_path, 'examples/files/LILA_collections/missouri_camera_traps_set1.json')
json_dets_path = os.path.join(
    conabio_ml_path, 'examples/files/detections/LILA/missouri-camera-traps_mdv5a.0.0_results.json')
images_dir = os.path.join(conabio_ml_path, 'examples/data/LILA_collections/missouri')

files_dir = './missouri_new_files'
os.makedirs(files_dir, exist_ok=True)
csv_path = os.path.join(files_dir, 'imgs_ds.csv')
crops_dir = os.path.join(files_dir, 'crop_images')
crops_ds_file = os.path.join(files_dir, 'crops_ds.csv')

if not os.path.isfile(crops_ds_file):
    ds_csv = ImageDataset.from_csv(csv_path)
    ds_json = ImageDataset.from_json(json_path, collection='missouri')

    # Se crea columna temporal para conectar ds_json y ds_csv
    ds_json.set_images_info_field_by_expr(
        field='item_csv',
        values_expr=lambda row: os.path.join(row['file_name'].split('\\')[1],
                                            row['file_name'].split('\\')[-1]),
        inplace=True)

    # Dejar en ds_json solo los elementos que hay en ds_csv
    ds_json.filter_by_column('item_csv', ds_csv.get_unique_items(), mode='include', inplace=True)

    dets_ds = ImagePredictionDataset.from_json(json_dets_path, validate_filenames=False)
    dets_ds.set_data_field_by_expr(                 # Se quita 'images/' al inicio de 'item'
        field='item',
        values_expr=lambda row: '/'.join(row['item'].split('/')[1:]),
        inplace=True)
    # Se modifica el campo 'image_id' de ds_json para que coincida con el de dets_ds
    ds_json.set_images_info_field_by_expr(
        field='image_id', values_expr=lambda row: Path(row['item_csv']).stem, inplace=True)

    dets_ds.filter_by_column('item', ds_json.get_unique_items(), mode='include', inplace=True)
    dets_ds.only_first_prediction()
    # Se convierten las coordenadas de formato relativo a formato absoluto (p.ej. 0.123 -> 234 px)
    dets_ds.set_images_dir(images_dir)
    dets_ds.convert_coordinates(
        input_format=coords_utils.COORDINATES_FORMATS.X_Y_WIDTH_HEIGHT,
        output_format=coords_utils.COORDINATES_FORMATS.X_Y_WIDTH_HEIGHT,
        output_coords_type=coords_utils.COORDINATES_TYPES.ABSOLUTE)

    ds_w_bboxes = create_obj_level_ds_from_dets_and_anns(detections=dets_ds, annotations=ds_json)
    ds_w_bboxes.set_images_dir(images_dir)
    crops_ds = ds_w_bboxes.create_classif_ds_from_bboxes_crops(dest_path=crops_dir)
    crops_ds.to_folder(crops_dir, keep_originals=False, split_in_labels=True, set_images_dir=True)
    crops_ds.to_csv(crops_ds_file, remove_label_to_item_path=True)
else:
    crops_ds = ImageDataset.from_csv(
        crops_ds_file, images_dir=crops_dir, append_label_to_item_path=True)
