import os

# region constanst
RAND_STATE = 1998
# endregion

min_score_detections = 0.4
azcopy_batch_size = 5000

files_dir = os.path.join('files')
metadata_dir = os.path.join('metadatadir')
azcopy_dir = os.path.join(files_dir, 'azcopy_exec')


os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(files_dir, exist_ok=True)
os.makedirs(azcopy_dir, exist_ok=True)

mapping_classes_csv = os.path.join(files_dir, 'lila-taxonomy-mapping_release.csv')
md_results_paths = os.path.join(files_dir, 'LILA_collections_MDv5a_MDv5b_files.csv')
azcopy_exec = os.path.join(azcopy_dir, 'azcopy')

results_dir = os.path.join('results', 'create_lila_ds')

#lila_imgs_csv = os.path.join(results_dir, f'datasets-imgs', f'Missouri_Camera_Traps.csv')
#lila_bboxes_csv = os.path.join(results_dir, f'datasets-bboxes', f'Missouri_Camera_Traps.csv')
#os.makedirs(os.path.dirname(lila_imgs_csv), exist_ok=True)
#os.makedirs(os.path.dirname(lila_bboxes_csv), exist_ok=True)


# region results and common files
results_path = os.path.join('results')
common_files_path = os.path.join('..', 'files')
local_files_path = os.path.join('files')
data_path = os.path.join('data', 'animals')
# endregion

# region images folders

lila_images_dir = os.path.join(data_path, 'images', 'lila')
lila_crops_dir = os.path.join(data_path, 'crops', 'lila')
lila_crops_split_dir = os.path.join(data_path, 'crops_split', 'lila')

os.makedirs(lila_images_dir , exist_ok=True)
os.makedirs(lila_crops_dir, exist_ok=True)
os.makedirs(lila_crops_split_dir, exist_ok=True)
# endregion

# region results files

lila_dataset_csv = os.path.join(results_path,'datasets', 'LILA','Missouri_Camera_Traps.csv')
lila_crops_ds_csv = os.path.join(results_path, 'datasets', 'LILA', f'crops_ds.csv')
crops_split_ds_csv = os.path.join(results_path, 'datasets', 'LILA', f'crops_split_ds.csv')
# endregion

os.makedirs(os.path.dirname(lila_dataset_csv), exist_ok = True)
os.makedirs(os.path.dirname(lila_crops_ds_csv), exist_ok = True)
os.makedirs(os.path.dirname(crops_split_ds_csv), exist_ok = True)









