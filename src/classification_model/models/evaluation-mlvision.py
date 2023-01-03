import os
import argparse

from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3

from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns
from conabio_ml_vision.datasets.datasets import ImageDataset, ConabioImageDataset
from conabio_ml_vision.utils.aux_utils import get_spcs_of_int
from conabio_ml_vision.utils.evaluator_utils import get_image_level_binary_pred_dataset
from conabio_ml_vision.utils.aux_utils import eval_multiclass, eval_binary, balance_binary_dataset
from conabio_ml_vision.utils.evaluator_utils import precision_recall_curve
from conabio_ml.utils.utils import str2bool

from utils import *


def train_model(model_names, binary_model):
    model_type = 'binary' if binary_model else 'multiclass'
    crops_ds_file = os.path.join(obj_level_datasets_path, f'all.csv')
    crops_imgs_dir = os.path.join(crops_imgs_path, 'experiment_4_train_test')
    dataset = ImageDataset.from_csv(
        crops_ds_file, images_dir=crops_imgs_dir,
        append_partition_to_item_path=True, append_label_to_item_path=True,
        parallel_set_images_dir=True)
    if binary_model:
        dataset.map_categories(mapping_classes=binary_mappings)
    for model_name in model_names:
        train(model_checkpoint_path=model_ckpts[model_type][model_name],
              dataset=dataset,
              labelmap_path=labelmap_files[model_type],
              model_type=model_base_instances[model_name],
              model_name=model_name,
              images_size=model_input_sizes[model_name],
              epochs=N_EPOCHS,
              batch_size=BATCH_SIZE_TRAIN)


def classify_and_eval_model(model_names,
                            binary_model,
                            only_snmb,
                            wo_clahe,
                            partition=Partitions.TEST):
    model_type = 'binary' if binary_model else 'multiclass'
    max_classifs = 2 if binary_model else 10
    clahe_str = '-wo_clahe' if wo_clahe else ''
    if only_snmb:
        crops_ds_file = os.path.join(obj_level_datasets_path, f'snmb{clahe_str}.csv')
        crops_imgs_dir = os.path.join(crops_imgs_path, f'snmb{clahe_str}')
    else:
        crops_ds_file = os.path.join(obj_level_datasets_path, f'all.csv')
        crops_imgs_dir = os.path.join(crops_imgs_path, 'experiment_4_train_test')

    dataset = ImageDataset.from_csv(
        source_path=crops_ds_file, images_dir=crops_imgs_dir,
        append_partition_to_item_path=True, append_label_to_item_path=True,
        validate_filenames=False)
    dataset.filter_by_partition(partition)
    for model_name in model_names:
        classifs_csv = os.path.join(
            classifs_base_path, model_type, f"{partition}{clahe_str}_{model_name}.csv")
        preds_model_ds = classify_dataset(
            dataset=dataset,
            classifs_csv=classifs_csv,
            labelmap_path=labelmap_files[model_type],
            model_checkpoint=model_ckpts[model_type][model_name],
            images_size=model_input_sizes[model_name],
            batch_size=BATCH_SIZE_EVAL,
            partition=partition,
            max_classifs=max_classifs)
        if model_type == 'multiclass':
            ev_dir = os.path.join(
                evals_base_path, "obj_level", 'multiclass', f"{partition}{clahe_str}_{model_name}")
            title = (
                f'Multiclass object-level evaluation with the model {model_name} on the\n'
                f'{partition} partition of the {"SNMB" if only_snmb else "Experiment-4"} dataset')
            eval_multiclass(
                dataset_true=dataset,
                dataset_pred=preds_model_ds,
                eval_dir=ev_dir,
                partition=partition,
                title=title,
                sample_counts_in_bars=True)
    dest_classifs_csv = os.path.join(
        classifs_base_path, model_type, f"{partition}_ensemble{clahe_str}.csv")
    source_classifs_csv_base = os.path.join(
        classifs_base_path, model_type, f"{partition}{clahe_str}.csv")
    preds_ensemble_ds = ensemble_classification_models(
        models_names=models_to_ensemble,
        dest_classifs_csv=dest_classifs_csv,
        images_dir=crops_imgs_dir,
        source_classifs_csv_base=source_classifs_csv_base,
        model_weights=model_weights)
    if model_type == 'multiclass':
        title = (
            f'Multiclass object-level evaluation with the ensemble of models '
            f'{", ".join(model_names)}\non the {partition} partition of the '
            f'{"SNMB" if only_snmb else "Experiment-4"} dataset')
        eval_dir = os.path.join(
            evals_base_path, "obj_level", 'multiclass', f"{partition}_ensemble{clahe_str}")
        eval_multiclass(
            dataset_true=dataset,
            dataset_pred=preds_ensemble_ds,
            partition=partition,
            eval_dir=eval_dir,
            title=title,
            sample_counts_in_bars=True)


def evaluation_binary_obj_level(binary_model):
    model_type = 'binary' if binary_model else 'multiclass'
    crops_ds_file = os.path.join(obj_level_datasets_path, f'all.csv')
    crops_imgs_dir = os.path.join(crops_imgs_path, 'experiment_4_train_test')
    evals_base_dir = os.path.join(evals_base_path, "obj_level", "binary", f"{model_type}_model")

    dataset_true = ImageDataset.from_csv(
        crops_ds_file, images_dir=crops_imgs_dir,
        append_partition_to_item_path=True, append_label_to_item_path=True)
    dataset_true.map_categories(mapping_classes=binary_mappings)
    df_true = dataset_true.as_dataframe()
    perc_empty = int(len(df_true[df_true.label == 'empty']) / len(df_true) * 100)
    dataset_pred = ImagePredictionDataset.from_csv(
        source_path=os.path.join(classifs_base_path, model_type, f"test_ensemble.csv"),
        images_dir=crops_imgs_dir)
    dataset_pred.only_first_prediction()
    if model_type == 'multiclass':
        dataset_pred.map_categories(binary_mappings)

    evals_results = []
    for score_thrs in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thrs, 2)*100)}'
        dataset_pred_thres = dataset_pred.copy().set_data_field_by_expr(
            'label',
            lambda x: 'animal' if x['label'] == 'animal' and x['score'] >= score_thrs else 'empty')
        title = (
            f'Binary object-level evaluation with a {model_type} model on the Test partition of\n'
            f'the Experiment-4 dataset ({perc_empty}% empty crops) for animal class and\n'
            f'a threshold {score_thrs:.2f}')
        res_eval = eval_binary(
            dataset_true=dataset_true,
            dataset_pred=dataset_pred_thres,
            eval_dir=os.path.join(evals_base_dir, f"thres_{thres_str}"),
            partition=Partitions.TEST,
            title=title,
            sample_counts_in_bars=True)
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    title = (
        f'Precision-Recall Curve for object-level binary evaluation with a {model_type} model on\n'
        f'the Test partition of the Experiment-4 dataset ({perc_empty}% empty crops) for\n'
        f'animal class')
    precision_recall_curve(evals_results, evals_base_dir, title=title)


def evaluation_binary_image_level(only_megadetector, only_snmb, binary_model, wo_clahe, perc_animal=None):
    model_type = 'binary' if binary_model else 'multiclass'
    clahe_str = '-wo_clahe' if wo_clahe else ''
    folder_name = "megadetector"
    folder_name += f"-{'wo' if only_megadetector else model_type}_model"
    folder_name += f"-{'snmb' if only_snmb else 'complete'}_dataset"
    folder_name += clahe_str
    if perc_animal is not None:
        folder_name += f"-perc_animal_{int(perc_animal*100)}"
    eval_path = os.path.join(evals_base_path, "image_level", "binary", folder_name)
    perc_animal = perc_animal if perc_animal is not None else 0.20

    if only_snmb:
        img_level_ds_file = os.path.join(img_level_datasets_path, f'snmb.csv')
        animal_dets_csv = snmb_animal_dets_wo_clahe_csv if wo_clahe else snmb_animal_dets_csv
        empty_dets_csv = snmb_empty_dets_wo_clahe_csv if wo_clahe else snmb_empty_dets_csv
        snmb_animal_dets_ds = ImagePredictionDataset.from_csv(animal_dets_csv)
        snmb_empty_dets_ds = ImagePredictionDataset.from_csv(empty_dets_csv)
        dets_ds = ImagePredictionDataset.from_datasets(
            snmb_animal_dets_ds, snmb_empty_dets_ds, round_score_digits=DEC_DIGITS_RND_SCR)
    else:
        img_level_ds_file = os.path.join(img_level_datasets_path, f'all.csv')
        det_animal_dfs = [pd.read_csv(x)
                          for x in dets_animal_colls_path.values() if os.path.isfile(x)]
        det_empty_dfs = [pd.read_csv(x)
                         for x in dets_empty_colls_path.values() if os.path.isfile(x)]
        dets_ds = ImagePredictionDataset.from_dataframes(
            det_animal_dfs + det_empty_dfs, round_score_digits=DEC_DIGITS_RND_SCR)

    if only_megadetector == False:
        classified_crops_ds = ImagePredictionDataset.from_csv(
            os.path.join(classifs_base_path, model_type, f"test_ensemble{clahe_str}.csv"))
        if model_type == 'multiclass':
            classified_crops_ds.map_categories(mapping_classes=binary_mappings)

    true_img_level_ds = ImageDataset.from_csv(img_level_ds_file)
    if not only_snmb:
        true_img_level_ds.filter_by_partition('test')
    true_img_level_ds.map_categories(mapping_classes=binary_mappings)

    if only_snmb:
        balance_binary_dataset(true_img_level_ds, perc_animal, by_partition=True)
    true_img_level_df = true_img_level_ds.as_dataframe()
    perc_empty = int(len(true_img_level_df[true_img_level_df.label == 'empty']) /
                     len(true_img_level_df) * 100)
    title_str = (
        f'with Megadetector {("and a " + model_type + " model") if not only_megadetector else ""}\n'
        f'on the Test partition of {"the SNMB portion of images of " if only_snmb else ""}'
        f'the Experiment-4 dataset ({perc_empty}% empty images)\nfor animal class')

    evals_results = []
    for score_thres in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thres, 2)*100)}'
        dets_ds_thres = dets_ds.copy().filter_by_score(min_score=score_thres)

        if only_megadetector:
            pred_ds_thres = dets_ds_thres
        else:
            dets_ids_thres = dets_ds_thres.as_dataframe()['id'].values
            classified_crops_ds_thres = classified_crops_ds.copy().filter_by_column(
                column='id', values=dets_ids_thres)
            classified_crops_ds_thres.only_first_prediction()
            classified_crops_ds_thres.filter_by_categories('empty', mode='exclude')
            pred_ds_thres = classified_crops_ds_thres

        dataset_pred = get_image_level_binary_pred_dataset(true_img_level_ds, pred_ds_thres)
        res_eval = eval_binary(
            dataset_true=true_img_level_ds,
            dataset_pred=dataset_pred,
            eval_dir=os.path.join(eval_path, f"thres_{thres_str}"),
            partition=None,
            title=f'Binary image-level evaluation {title_str} and a threshold {score_thres:.2f}',
            sample_counts_in_bars=True)
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    title = f'Precision-Recall Curve for image-level binary evaluation {title_str}'
    precision_recall_curve(evals_results, plot_path=eval_path, title=title)


def evaluation_multiclass_image_level(only_snmb,
                                      functional_groups,
                                      threshold=None,
                                      partition=Partitions.TEST):
    perc_animal = 0.20
    folder_name = "functional_groups" if functional_groups else "species"
    folder_name += "-snmb_dataset" if only_snmb else "-complete_dataset"
    folder_name += f"-{partition}"
    eval_path = os.path.join(evals_base_path, "image_level", "multiclass", folder_name)

    classif_crops_csv = os.path.join(classifs_base_path, 'multiclass', f"{partition}_ensemble.csv")
    classif_crops_ds = ImagePredictionDataset.from_csv(classif_crops_csv)
    img_level_csv = os.path.join(img_level_datasets_path, f'{"snmb" if only_snmb else "all"}.csv')
    true_img_level_ds = ImageDataset.from_csv(img_level_csv)
    true_img_level_ds.filter_by_partition(partition)

    if only_snmb:
        balance_binary_dataset(
            true_img_level_ds, perc_animal, by_partition=False, random_state=RANDOM_STATE)

    labels = None
    if functional_groups:
        true_img_level_ds.map_categories(mappings_species_to_fgs)
        classif_crops_ds.map_categories(mappings_species_to_fgs)
        labels = list(set(true_img_level_ds.get_categories()) - {'gallus gallus', 'empty'})

    true_img_level_df = true_img_level_ds.as_dataframe()

    perc_empty = int(
        len(true_img_level_df[true_img_level_df.label == 'empty']) / len(true_img_level_df) * 100)
    title_str = (
        f'with Megadetector and a multiclass model\n'
        f'on the Test partition of {"the SNMB portion of images of " if only_snmb else ""}'
        f'the Experiment-4 dataset ({perc_empty}% empty images)\n')

    evals_results = []
    thresholds = ([threshold] if threshold is not None
                  else np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS))
    for score_thres in thresholds:
        thres_str = f'{int(round(score_thres, 2)*100)}'
        pred_ds_thres = classif_crops_ds.copy().filter_by_score(
            min_score=score_thres, column_name="score_det")
        pred_ds_thres.only_first_prediction()
        pred_ds_thres.filter_by_categories('empty', mode='exclude')

        pred_img_level_ds = get_img_level_multiclass_preds(true_img_level_ds, pred_ds_thres)

        eval_path_thres = os.path.join(eval_path, f"thres_{thres_str}")
        title = f'Multiclass image-level evaluation {title_str} and a threshold {score_thres:.2f}'
        res_eval = eval_multiclass(
            dataset_true=true_img_level_ds,
            dataset_pred=pred_img_level_ds,
            eval_dir=eval_path_thres,
            partition=None,
            title=title,
            labels=labels,
            sample_counts_in_bars=True)
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})

    title = f'Precision-Recall Curve for image-level multiclass evaluation {title_str}'
    precision_recall_curve(
        evals_results, plot_path=eval_path, title=title, multiclass_eval=True)


def analysis_by_location(threshold=None, perc_animal=None):
    res_path = os.path.join(results_path, 'analysis_by_location')

    classif_crops_train_csv = os.path.join(classifs_base_path, 'multiclass', f"train_ensemble.csv")
    classif_crops_train_ds = ImagePredictionDataset.from_csv(classif_crops_train_csv)
    classif_crops_test_csv = os.path.join(classifs_base_path, 'multiclass', f"test_ensemble.csv")
    classif_crops_test_ds = ImagePredictionDataset.from_csv(classif_crops_test_csv)
    classif_crops_ds = ImagePredictionDataset.from_datasets(
        classif_crops_train_ds, classif_crops_test_ds)

    img_level_csv = os.path.join(img_level_datasets_path, 'snmb.csv')
    true_img_level_ds = ImageDataset.from_csv(img_level_csv)
    
    if perc_animal is not None:
        balance_binary_dataset(
            true_img_level_ds, perc_animal, by_partition=False, random_state=RANDOM_STATE)
        res_path += f'-{int(round(perc_animal, 2)*100)}_perc_animal'

    thresholds = ([threshold] if threshold is not None
                  else np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS))
    for score_thres in thresholds:
        thres_str = f'{int(round(score_thres, 2)*100)}'

        path_thres = os.path.join(res_path, f"thres_{thres_str}")
        pred_ds_thres = classif_crops_ds.copy().filter_by_score(
            min_score=score_thres, column_name="score_det")
        pred_ds_thres.only_first_prediction()
        pred_ds_thres.filter_by_categories('empty', mode='exclude')

        pred_img_level_ds = get_img_level_multiclass_preds(
            true_img_level_ds, pred_ds_thres, inherit_fields=['partition'])

        by_location_analysis(true_img_level_ds, pred_img_level_ds, path_thres)


def eval_compare_w_kale_pipeline(model_names):
    img_level_ds_file = os.path.join(img_level_datasets_path, 'all_test_part.csv')
    detections_csv = os.path.join(dets_datasets_path, 'all_test_part.csv')
    test_imgs_path = os.path.join(complete_imgs_path, 'experiment_4_test_complete_imgs_test_part')
    MIN_SCORE_DETS = 0.01

    # 1. Take the test partition from the dataset described in the previous section to create a true
    #    binary dataset, mapping the empty category to non-fauna and the rest of the
    #    categories (species) to fauna.
    binary_true_ds = ImageDataset.from_csv(
        img_level_ds_file, images_dir=test_imgs_path, validate_filenames=False)
    binary_true_ds.map_categories(mapping_classes={'empty': 'non_fauna', '*': 'fauna'})

    # 2. Generate the detections with Megadetector on the dataset generated in 1.
    dets_ds = ImagePredictionDataset.from_csv(
        source_path=detections_csv, round_score_digits=DEC_DIGITS_RND_SCR)
    dets_ds.filter_by_categories([ConabioImageDataset.ANIMAL_LABEL_LWR])

    # 3. Make a cycle varying the threshold from 0 to 1 in steps of 0.05. For each step create a binary
    #    prediction dataset using the detections generated in 2, considering as fauna the photos with
    #    at least one animal detection with score ≥ threshold and as non-fauna the rest, and make the
    #    binary evaluation of each of these binary prediction datasets with respect to the true binary
    #    dataset of 1.
    folder_name = 'megadetector-wo_model-test_part'
    eval_path = os.path.join(evals_base_path, "image_level", "binary", folder_name)
    evals_results = []
    for score_thres in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thres, 2)*100)}'

        dets_ds_thres = dets_ds.copy().filter_by_score(min_score=score_thres)
        binary_pred_ds = get_image_level_binary_pred_dataset(
            binary_true_ds, dets_ds_thres, animal_label='fauna', empty_label='non_fauna')

        res_eval = eval_binary(
            dataset_true=binary_true_ds,
            dataset_pred=binary_pred_ds,
            eval_dir=os.path.join(eval_path, f"thres_{thres_str}"),
            partition=None,
            pos_label='fauna',
            sample_counts_in_bars=True,
            title=f'Binary image-level evaluation of Megadetector for a threshold {score_thres:.2f}')
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    precision_recall_curve(
        evals_results,
        plot_path=os.path.join(plots_path, 'precision_recall-megadet_new_dets.png'),
        title=(f'Precision-recall curve for binary evaluation at image level of the Megadetector '
               f'(full size images)'))

    # 4. Clip the bounding boxes of the detections generated in 2 and create a crops dataset
    crops_ds_path = os.path.join(obj_level_datasets_path, 'all_test_part.csv')
    crops_imgs_dir = os.path.join(crops_imgs_path, 'experiment_4_test')
    if not os.path.isfile(crops_ds_path):
        obj_level_ds = create_obj_level_ds_from_dets_and_anns(
            detections=dets_ds,
            annotations=binary_true_ds,
            det_categories=[ConabioImageDataset.ANIMAL_LABEL_LWR],
            include_detection_score=True,
            min_score_detections=MIN_SCORE_DETS,
            validate_filenames=False)
        crops_ds = obj_level_ds.create_classif_ds_from_bboxes_crops(
            dest_path=crops_imgs_dir, allow_label_empty=True)
        crops_ds.to_csv(crops_ds_path)
    else:
        crops_ds = ImageDataset.from_csv(
            crops_ds_path, images_dir=crops_imgs_dir, validate_filenames=False)

    # 5. Apply the classification of the multiclass model (Rekognition or EfficientNet) to the crops
    #    dataset generated in 4.
    for model_name in model_names:
        classifs_csv = os.path.join(
            classifs_base_path, 'multiclass_new_dets', f"test_{model_name}.csv")
        _ = classify_dataset(
            dataset=crops_ds,
            classifs_csv=classifs_csv,
            labelmap_path=labelmap_files['multiclass'],
            model_checkpoint=model_ckpts['multiclass'][model_name],
            images_size=model_input_sizes[model_name],
            batch_size=BATCH_SIZE_EVAL,
            partition=None,
            max_classifs=10)
    dest_classifs_csv = os.path.join(
        classifs_base_path, 'multiclass_new_dets', f"test_ensemble.csv")
    source_classifs_csv_base = os.path.join(classifs_base_path, 'multiclass_new_dets', f"test.csv")
    preds_ensemble_ds = ensemble_classification_models(
        models_names=models_to_ensemble,
        dest_classifs_csv=dest_classifs_csv,
        images_dir=crops_imgs_dir,
        source_classifs_csv_base=source_classifs_csv_base,
        model_weights=model_weights)

    # 6. Repeat step 3, previously discarding the detections whose crops have been classified
    #    as empty in 5.
    preds_ensemble_ds.filter_by_categories('empty', mode='exclude')
    evals_results = []
    for score_thres in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thres, 2)*100)}'

        dets_ds_thres = dets_ds.copy().filter_by_score(min_score=score_thres)
        dets_ids_thres = dets_ds_thres.as_dataframe()['id'].values
        classified_crops_ds_thres = preds_ensemble_ds.copy().filter_by_column(
            column='id', values=dets_ids_thres)
        binary_pred_ds = get_image_level_binary_pred_dataset(
            binary_true_ds, classified_crops_ds_thres,
            animal_label='fauna', empty_label='non_fauna')

        title = (f'Binary image-level evaluation of Megadetector + EfficientNet ensemble '
                 f'for a threshold {score_thres:.2f}')
        eval_dir = os.path.join(
            evals_base_path, 'image_level', "megadetector_efficientnet", f"thres_{thres_str}")
        res_eval = eval_binary(
            dataset_true=binary_true_ds,
            dataset_pred=binary_pred_ds,
            eval_dir=eval_dir,
            partition=None,
            pos_label='fauna',
            sample_counts_in_bars=True,
            title=title)
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    precision_recall_curve(
        evals_results,
        plot_path=os.path.join(plots_path, 'precision_recall-megadet_efficientnet.png'),
        title=(f'Precision-Recall Curve for binary image-level evaluation '
               f'of Megadetector + EfficientNet ensemble'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--create_dataset", default=False, action="store_true")
    parser.add_argument("--train_model", default=False, action="store_true")
    parser.add_argument("--classify_and_eval_model", default=False, action="store_true")
    parser.add_argument("--evaluation_binary_obj_level", default=False, action="store_true")
    parser.add_argument("--evaluation_binary_image_level", default=False, action="store_true")
    parser.add_argument("--evaluation_multiclass_image_level", default=False, action="store_true")
    parser.add_argument("--evaluation_binary_img_level_spcs", default=False, action="store_true")
    parser.add_argument("--eval_compare_w_kale_pipeline", default=False, action="store_true")
    parser.add_argument("--analysis_by_location", default=False, action="store_true")

    parser.add_argument('--score_thres_animal', default=0.7, type=float)
    parser.add_argument('--score_thres_empty', default=0.3, type=float)
    parser.add_argument('--collections', default='all', type=str)
    parser.add_argument('--train_percent', default=0.8, type=float)
    parser.add_argument('--perc_animal', default=None, type=float)
    parser.add_argument('--model', choices=['EfficientNetB0', 'EfficientNetB3'])
    parser.add_argument("--only_megadetector", default=False, action="store_true")
    parser.add_argument("--only_snmb", default=False, action="store_true")
    parser.add_argument("--binary_model", default=False, action="store_true")
    parser.add_argument("--wo_clahe", default=False, action="store_true")
    parser.add_argument("--functional_groups", default=False, action="store_true")
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--partition', default=Partitions.TEST,
                        choices=[Partitions.TEST, Partitions.TRAIN])
    parser.add_argument('--splitting_type', choices=['random', 'stratified', 'old_stratified'])
    parser.add_argument("--validate_filenames",
                        type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()

    results_path = os.path.join('..', 'results', f'eval_megadet_rekog_exp_4')
    if args.splitting_type is not None:
        results_path += '-split'
        results_path = os.path.join(results_path, args.splitting_type)
    datasets_path = os.path.join(results_path, 'datasets')
    img_level_datasets_path = os.path.join(datasets_path, 'image_level')
    obj_level_datasets_path = os.path.join(datasets_path, 'obj_level')
    dets_datasets_path = os.path.join(datasets_path, 'detections')
    plots_path = os.path.join(datasets_path, 'plots')
    imgs_path = os.path.join(results_path, 'images')
    complete_imgs_path = os.path.join(imgs_path, 'complete')
    crops_imgs_path = os.path.join(imgs_path, 'crops')
    models_path = os.path.join(results_path, 'models')
    classifs_base_path = os.path.join(results_path, 'classifs')
    evals_base_path = os.path.join(results_path, 'evals')

    os.makedirs(img_level_datasets_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    # Training params
    labelmap_files = {
        'multiclass': os.path.join(models_path, 'multiclass', f"labels.txt"),
        'binary': os.path.join(models_path, 'binary', f"labels.txt")
    }
    model_ckpts = {
        'multiclass': {
            'EfficientNetB0': os.path.join(models_path, 'multiclass', 'efficientnet_b0.model.hdf5'),
            'EfficientNetB3': os.path.join(models_path, 'multiclass', 'efficientnet_b3.model.hdf5')
        },
        'binary': {
            'EfficientNetB0': os.path.join(models_path, 'binary', 'efficientnet_b0.model.hdf5'),
            'EfficientNetB3': os.path.join(models_path, 'binary', 'efficientnet_b3.model.hdf5')
        }
    }
    model_base_instances = {
        'EfficientNetB0': EfficientNetB0,
        'EfficientNetB3': EfficientNetB3
    }
    # Training params
    model_input_sizes = {
        'EfficientNetB0': (224, 224),
        'EfficientNetB3': (300, 300),
    }
    model_weights = {
        'EfficientNetB0': .4,
        'EfficientNetB3': .6,
    }
    models_to_ensemble = list(model_ckpts['multiclass'].keys())
    N_EPOCHS = 20
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL = 16

    model_names = models_to_ensemble if args.model is None else args.model.split(',')

    if args.create_dataset:
        create_dataset(
            collections=args.collections,
            score_thres_animal=args.score_thres_animal,
            score_thres_empty=args.score_thres_empty,
            train_percent=args.train_percent,
            wo_clahe=args.wo_clahe,
            splitting_type=args.splitting_type,
            validate_filenames=args.validate_filenames)
    if args.train_model:
        train_model(model_names=model_names, binary_model=args.binary_model)
    if args.classify_and_eval_model:
        classify_and_eval_model(
            model_names=model_names,
            binary_model=args.binary_model,
            only_snmb=args.only_snmb,
            wo_clahe=args.wo_clahe,
            partition=args.partition)
    if args.evaluation_binary_obj_level:
        evaluation_binary_obj_level(binary_model=args.binary_model)
    if args.evaluation_binary_image_level:
        evaluation_binary_image_level(
            args.only_megadetector,
            args.only_snmb,
            binary_model=args.binary_model,
            wo_clahe=args.wo_clahe,
            perc_animal=args.perc_animal)
    if args.evaluation_multiclass_image_level:
        evaluation_multiclass_image_level(
            only_snmb=args.only_snmb,
            functional_groups=args.functional_groups,
            threshold=args.threshold,
            partition=args.partition)
    if args.eval_compare_w_kale_pipeline:
        eval_compare_w_kale_pipeline(model_names=model_names)
    if args.analysis_by_location:
        analysis_by_location(threshold=args.threshold, perc_animal=args.perc_animal)
