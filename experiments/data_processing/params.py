import os

files_dir = os.path.join('files')
metadata_dir = os.path.join('metadatadir')
mapping_classes_csv = os.path.join(files_dir, 'lila-taxonomy-mapping_release.csv')
md_results_paths = os.path.join(files_dir, 'LILA_collections_MDv5a_MDv5b_files.csv')

azcopy_dir = os.path.join(files_dir, 'azcopy_exec')
azcopy_exec = os.path.join(azcopy_dir, 'azcopy')

results_dir = os.path.join('results', 'create_complete_lila_ds')

lila_imgs_csv = os.path.join(results_dir, f'datasets-imgs', f'lila_complete.csv')
lila_bboxes_csv = os.path.join(results_dir, f'datasets-bboxes', f'lila_complete.csv')

sampled_ds_csv = os.path.join(results_dir, 'sampled_ds.csv')
crops_ds_csv = os.path.join(results_dir, 'crops_ds.csv')
dets_of_sampled_ds_csv = os.path.join(results_dir, 'dets_of_sampled_ds.csv')
os.makedirs(os.path.dirname(lila_imgs_csv), exist_ok=True)
os.makedirs(os.path.dirname(lila_bboxes_csv), exist_ok=True)


os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(files_dir, exist_ok=True)
os.makedirs(azcopy_dir, exist_ok=True)


CLASS_TAXA_COL = 'class_taxa'
COLLECTION_COL = 'collection'

RND_STATE = None
min_score_detections = 0.4
azcopy_batch_size = 5000
