import os
from pathlib import Path

from conabio_ml_vision.utils import coords_utils
from conabio_ml_vision.datasets import ImageDataset, ImagePredictionDataset
from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns


if __name__ == '__main__':
    __spec__ = None

    json_path = os.path.join( '../../../data/files/LILA_collections/missouri_camera_traps_set1.json')
    json_dets_path = os.path.join( '../../../data/files/LILA_collections/missouri-camera-traps_mdv5a.0.0_results.json')
    csv_dets_path = os.path.join( '../../../data/files/LILA_collections/missouri-camera-traps_mdv5a.0.0_results.csv')

    images_dir = '../../../data/raw'
    files_dir = '../../../data/files/missouri_new_files'
    csv_path = os.path.join(files_dir, 'imgs_ds.csv')
    crops_dir = os.path.join(files_dir, 'crop_images')
    crops_ds_file = os.path.join(files_dir, 'crops_ds.csv')
    imgs_ds_file = os.path.join(files_dir, 'images_ds.csv')
    dets_ds_file = os.path.join(files_dir, 'dets_ds.csv')

    ds_json = ImageDataset.from_json(json_path,images_dir=images_dir,validate_filenames=True)
    ds_json.set_images_info_field_by_expr(
        field='item_csv',
        values_expr=lambda row: os.path.join(row['file_name'].split('\\')[1],
                                            row['file_name'].split('\\')[-1]),
        inplace=True)

    dets_ds = ImagePredictionDataset.from_json(json_dets_path, validate_filenames=False)
    dets_ds.set_data_field_by_expr(                 # Se quita 'images/' al inicio de 'item'
            field='item',
            values_expr=lambda row: '/'.join(row['item'].split('/')[1:]),
            inplace=True)
    # Se modifica el campo 'image_id' de ds_json para que coincida con el de dets_ds
    ds_json.set_images_info_field_by_expr(
        field='image_id', values_expr=lambda row: Path(row['item_csv']).stem, inplace=True)

    dets_ds.filter_by_column('item', ds_json.get_unique_items(), mode='include', inplace=True)

    dets_ds = ImagePredictionDataset.from_csv(csv_dets_path,images_dir=images_dir,validate_filenames=True)
    dets_ds.convert_coordinates(
                    input_format=coords_utils.COORDINATES_FORMATS.X_Y_WIDTH_HEIGHT,
                    output_format=coords_utils.COORDINATES_FORMATS.X_Y_WIDTH_HEIGHT,
                    output_coords_type=coords_utils.COORDINATES_TYPES.ABSOLUTE)
    
    dets_ds.only_first_prediction()
    #save results
    dets_ds.to_csv(dets_ds_file)
    ds_json.to_csv(imgs_ds_file)
    
    


    