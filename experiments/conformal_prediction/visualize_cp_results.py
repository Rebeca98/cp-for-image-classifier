import numpy as np
import os
import json
import argparse
import configparser
import random
# torch and torchvision
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import datasets
# conformal_classification
from conformal_classification.conformal import ConformalModel
from conformal_classification.utils_experiments import validate
from conformal_classification.utils_experiments import get_calib_transform
from conformal_classification.utils_experiments import build_model_for_cp
# plot libraries
import matplotlib.pyplot as plt
from paths import TEST_DIR, OUTPUT_DIR, MODELS_INFO, CAL_DIR, OUTPUT_CP_DIR
# import hashlib

if __name__ == "__main__":
    # paths
    # script_directory = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_directory)
    output_path = os.path.join("..", "results", "conformal_prediction")
    models_info = os.path.join("..", "files", "models_metadata.json")

    # config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help='Config file')
    args = parser.parse_args()
    # config_file_name = args.config_file.split("/")[1].split(".")[0]
    config_file_name = os.path.basename(args.config_file).split(".")[0]

    config = configparser.ConfigParser()
    config.read(args.config_file)

    defaults = {}
    defaults.update(dict(config.items("cp")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()  # Overwrite arguments
    # experiment arguments
    image_size = int(args.image_size)
    seed = int(args.random_seed)
    num_classes = int(args.num_classes)
    bsz = int(args.batch_size)
    # model_name = str(args.model_name)

    defaults.update(dict(config.items("raps_1_1")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_raps_1_1 = eval(args.predictor)
    kreg_raps_1_1 = eval(args.kreg)
    lamda_raps_1_1 = eval(args.lamda)
    randomized_raps_1_1 = eval(args.randomized)
    pct_paramtune_raps_1_1 = float(args.pct_paramtune)
    lamda_criterion_raps_1_1 = eval(args.lamda_criterion)
    strata_raps_1_1 = eval(args.strata)
    allow_zero_sets_raps_1_1 = eval(args.allow_zero_sets)
    alpha_raps_1_1 = eval(args.alpha)

    defaults.update(dict(config.items("raps_1_5")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_raps_1_5 = eval(args.predictor)
    kreg_raps_1_5 = eval(args.kreg)
    lamda_raps_1_5 = eval(args.lamda)
    randomized_raps_1_5 = eval(args.randomized)
    pct_paramtune_raps_1_5 = float(args.pct_paramtune)
    lamda_criterion_raps_1_5 = eval(args.lamda_criterion)
    strata_raps_1_5 = eval(args.strata)
    allow_zero_sets_raps_1_5 = eval(args.allow_zero_sets)
    alpha_raps_1_5 = eval(args.alpha)

    defaults.update(dict(config.items("raps_2_1")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_raps_2_1 = eval(args.predictor)
    kreg_raps_2_1 = eval(args.kreg)
    lamda_raps_2_1 = eval(args.lamda)
    randomized_raps_2_1 = eval(args.randomized)
    allow_zero_sets_raps_2_1 = eval(args.allow_zero_sets)
    alpha_raps_2_1 = eval(args.alpha)

    defaults.update(dict(config.items("raps_2_5")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_raps_2_5 = eval(args.predictor)
    kreg_raps_2_5 = eval(args.kreg)
    lamda_raps_2_5 = eval(args.lamda)
    randomized_raps_2_5 = eval(args.randomized)
    allow_zero_sets_raps_2_5 = eval(args.allow_zero_sets)
    alpha_raps_2_5 = eval(args.alpha)

    defaults.update(dict(config.items("aps_1")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_aps_1 = eval(args.predictor)
    kreg_aps_1 = eval(args.kreg)
    lamda_aps_1 = eval(args.lamda)
    randomized_aps_1 = eval(args.randomized)
    allow_zero_sets_aps_1 = eval(args.allow_zero_sets)
    alpha_aps_1 = eval(args.alpha)

    defaults.update(dict(config.items("aps_5")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_aps_5 = eval(args.predictor)
    kreg_aps_5 = eval(args.kreg)
    lamda_aps_5 = eval(args.lamda)
    randomized_aps_5 = eval(args.randomized)
    allow_zero_sets_aps_5 = eval(args.allow_zero_sets)
    alpha_aps_5 = eval(args.alpha)

    defaults.update(dict(config.items("naive_1")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_naive_1 = eval(args.predictor)
    kreg_naive_1 = eval(args.kreg)
    lamda_naive_1 = eval(args.lamda)
    randomized_naive_1 = eval(args.randomized)
    allow_zero_sets_naive_1 = eval(args.allow_zero_sets)
    alpha_naive_1 = eval(args.alpha)

    defaults.update(dict(config.items("naive_5")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    predictor_naive_5 = eval(args.predictor)
    kreg_naive_5 = eval(args.kreg)
    lamda_naive_5 = eval(args.lamda)
    randomized_naive_5 = eval(args.randomized)
    allow_zero_sets_naive_5 = eval(args.allow_zero_sets)
    alpha_naive_5 = eval(args.alpha)

    # device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")

    # enable an automatic algorithm selection process within cuDNN
    cudnn.benchmark = True
    # Fix randomness
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    # JSON file that contains directories or paths to trained models.
    # results path
    f = open(MODELS_INFO)
    models_info_dict = json.load(f)
    f.close()
    # modelnames = models_info_dict['models'].keys()
    # models_information = [(models_info_dict['models'][name]['file_path'],
    #                    models_info_dict['models'][name]['architecture']) for name in modelnames]
    # for modelinfo in models_information:
    _model = 'config-4'
    model = build_model_for_cp(model_path=models_info_dict['models'][_model]['file_path'],
                               model_name=models_info_dict['models'][_model]['architecture'],
                               num_classes=num_classes).to(device)
    # Conformalize model
    transform = get_calib_transform(image_size)
    # calib_dataset = datasets.ImageFolder(CAL_DIR, transform)
    dataset = torchvision.datasets.ImageFolder(CAL_DIR, transform)

    # num_calib = len(calib_dataset)
    # VALID_SPLIT = 0.9  # 10% of data used for validation
    # num_val = int(num_calib*VALID_SPLIT)

    # calib_data, val_data = torch.utils.data.random_split(calib_dataset, [num_calib-num_val, num_val])

    # Initialize loaders
    # calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=bsz, shuffle=True, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=bsz, shuffle=True, pin_memory=True)

    # RAPS Conformalize model
    # conformal_model = ConformalModel(model,
    #                            calib_loader,
    #                            alpha = alpha,
    #                            lamda = lamda,
    #                            kreg = kreg,
    #                            randomized = randomized,
    #                            allow_zero_sets = allow_zero_sets)

    ######
    num_data = len(dataset)
    TEST_SPLIT = 0.7  # 20% of data will be used to conformalize the model
    num_test = int(num_data*TEST_SPLIT)
    VALID_SPLIT = 0.10  # 10% of data will be used to conformalize the model
    num_calib = num_data-num_test
    num_val = int(num_calib*VALID_SPLIT)

    calib_data, test_data = torch.utils.data.random_split(dataset,
                                                          [num_data-num_test, num_test])

    calib_data, val_data = torch.utils.data.random_split(calib_data,
                                                         [num_calib-num_val, num_val])

    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=bsz, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=bsz, shuffle=True)

    # Conformalize model: choose the propoert configuration (naive, aps, raps)
    raps_conformal_model_1_1 = ConformalModel(model=model,
                                              calib_loader=calib_loader,
                                              alpha=alpha_raps_1_1,
                                              kreg=kreg_raps_1_1,
                                              lamda=lamda_raps_1_1,
                                              randomized=randomized_raps_1_1,
                                              allow_zero_sets=allow_zero_sets_raps_1_1,
                                              pct_paramtune=pct_paramtune_raps_1_1,
                                              naive=False,
                                              batch_size=bsz,
                                              lamda_criterion=lamda_criterion_raps_1_1,
                                              strata=strata_raps_1_1
                                              )

    raps_conformal_model_1_5 = ConformalModel(model=model,
                                              calib_loader=calib_loader,
                                              alpha=alpha_raps_1_5,
                                              kreg=kreg_raps_1_5,
                                              lamda=lamda_raps_1_5,
                                              randomized=randomized_raps_1_5,
                                              allow_zero_sets=allow_zero_sets_raps_1_5,
                                              pct_paramtune=pct_paramtune_raps_1_5,
                                              naive=False,
                                              batch_size=bsz,
                                              lamda_criterion=lamda_criterion_raps_1_5,
                                              strata=strata_raps_1_5
                                              )

    raps_conformal_model_2_1 = ConformalModel(model=model,
                                              calib_loader=calib_loader,
                                              alpha=alpha_raps_2_1,
                                              kreg=kreg_raps_2_1,
                                              lamda=lamda_raps_2_1,
                                              randomized=randomized_raps_2_1,
                                              allow_zero_sets=allow_zero_sets_raps_2_1,
                                              naive=False,
                                              batch_size=bsz
                                              )

    raps_conformal_model_2_5 = ConformalModel(model=model,
                                              calib_loader=calib_loader,
                                              alpha=alpha_raps_2_5,
                                              kreg=kreg_raps_2_5,
                                              lamda=lamda_raps_2_5,
                                              randomized=randomized_raps_2_5,
                                              allow_zero_sets=allow_zero_sets_raps_2_5,
                                              naive=False,
                                              batch_size=bsz
                                              )

    aps_conformal_model_1 = ConformalModel(model=model,
                                           calib_loader=calib_loader,
                                           alpha=alpha_aps_1,
                                           kreg=kreg_aps_1,
                                           lamda=lamda_aps_1,
                                           randomized=randomized_aps_1,
                                           allow_zero_sets=allow_zero_sets_aps_1,
                                           naive=False,
                                           batch_size=bsz
                                           )
    aps_conformal_model_5 = ConformalModel(model=model,
                                           calib_loader=calib_loader,
                                           alpha=alpha_aps_5,
                                           kreg=kreg_aps_5,
                                           lamda=lamda_aps_5,
                                           randomized=randomized_aps_5,
                                           allow_zero_sets=allow_zero_sets_aps_5,
                                           naive=False,
                                           batch_size=bsz
                                           )
    naive_conformal_model_1 = ConformalModel(model=model,
                                             calib_loader=calib_loader,
                                             alpha=alpha_naive_1,
                                             kreg=kreg_naive_1,
                                             lamda=lamda_naive_1,
                                             randomized=randomized_naive_1,
                                             allow_zero_sets=allow_zero_sets_naive_1,
                                             naive=True,
                                             batch_size=bsz
                                             )
    naive_conformal_model_5 = ConformalModel(model=model,
                                             calib_loader=calib_loader,
                                             alpha=alpha_naive_5,
                                             kreg=kreg_naive_5,
                                             lamda=lamda_naive_5,
                                             randomized=randomized_naive_5,
                                             allow_zero_sets=allow_zero_sets_naive_5,
                                             naive=True,
                                             batch_size=bsz
                                             )

    ######
    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    # top1, top5, coverage, size = validate(val_loader, conformal_model, print_bool=True)
    print("Complete!")
    # visualize results
    num_images = 8
    explore_data, _ = torch.utils.data.random_split(val_data, [num_images, num_val-num_images])
    explore_loader = torch.utils.data.DataLoader(
        explore_data, batch_size=1, shuffle=True, pin_memory=True)
    dict_classes = dataset.class_to_idx

    model_dicts = []
    model_dict = {}
    labeldict = {}
    for name, number in dict_classes.items():
        labeldict[number] = name
    mosaiclist = []
    sets = []
    labels = []
    for i, (img, label) in enumerate(explore_loader):
        unnormalized_img = (img * torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))+torch.Tensor(
            [0.485, 0.456, 0.406]).view(-1, 1, 1)

        output, S = raps_conformal_model_1_1(img.to(device))
        model_dict[f"raps_conformal_model_1_1-{i}"] = [{'image': i,
                                                        'real-label': [labeldict[label[0].item()]],
                                                        'sets': [labeldict[s] for s in S[0]]}]

        output, S = raps_conformal_model_1_5(img.to(device))
        model_dict[f"raps_conformal_model_1_5-{i}"] = [{'image': i,
                                                        'real-label': [labeldict[label[0].item()]],
                                                        'sets': [labeldict[s] for s in S[0]]}]

        output, S = raps_conformal_model_2_1(img.to(device))
        model_dict[f"raps_conformal_model_2_1-{i}"] = [{'image': i,
                                                        'real-label': [labeldict[label[0].item()]],
                                                        'sets': [labeldict[s] for s in S[0]]}]

        output, S = raps_conformal_model_2_5(img.to(device))
        model_dict[f"raps_conformal_model_2_5-{i}"] = [{'image': i,
                                                        'real-label': [labeldict[label[0].item()]],
                                                        'sets': [labeldict[s] for s in S[0]]}]

        output, S = aps_conformal_model_1(img.to(device))
        model_dict[f"aps_conformal_model_1-{i}"] = [{'image': i,
                                                     'real-label': [labeldict[label[0].item()]],
                                                     'sets': [labeldict[s] for s in S[0]]}]

        output, S = aps_conformal_model_5(img.to(device))
        model_dict[f"aps_conformal_model_5-{i}"] = [{'image': i,
                                                     'real-label': [labeldict[label[0].item()]],
                                                     'sets': [labeldict[s] for s in S[0]]}]

        output, S = naive_conformal_model_1(img.to(device))
        model_dict[f"naive_conformal_model_-{i}"] = [{'image': i,
                                                      'real-label': [labeldict[label[0].item()]],
                                                      'sets': [labeldict[s] for s in S[0]]}]

        output, S = naive_conformal_model_5(img.to(device))
        model_dict[f"naive_conformal_model_5-{i}"] = [{'image': i,
                                                       'real-label': [labeldict[label[0].item()]],
                                                       'sets': [labeldict[s] for s in S[0]]}]
        model_dicts.append(model_dict)
        mosaiclist = mosaiclist + [unnormalized_img]

    mosaiclist = [mosaiclist[i][0] for i in range(len(mosaiclist))]
    # Open the file in "write" mode and append the label to it
    # _set = [labeldict[s] for s in S[0]]
    # sets = sets + [_set]
    # labels = labels + [label[0].item()]
    # labels = labels + [labeldict[label[0].item()]]
    grid = torchvision.utils.make_grid(mosaiclist)
    fig, ax = plt.subplots(
        figsize=(min(num_images, 9)*5, np.floor(num_images/9+1)*5))
    ax.imshow(grid.permute(1, 2, 0), interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    # Generate a random number
    # random_number = random.randint(0, 9999)

    # Create a string to hash that includes the random number
    # data = f"Hello, World! {random_number}"

    # Create a SHA-256 hash object
    # sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the data
    # sha256_hash.update(data.encode())

    # Get the hexadecimal representation of the hash
    # hashed_data = sha256_hash.hexdigest()[0:5]

    fig_results_path = os.path.join(
        OUTPUT_CP_DIR, f"{_model}", config_file_name, "explore_images.png")
    os.makedirs(os.path.dirname(fig_results_path), exist_ok=True)
    plt.savefig(fig_results_path)

    # Specify the file path
    file_path = os.path.join(os.path.dirname(fig_results_path), "cp-examples.json")

    dict_to_save = {'data': model_dicts}
    # Dump data to JSON file
    with open(file_path, "w") as json_file:
        json.dump(dict_to_save, json_file, indent=4)
    # generate a table with results
    # dfs = []
    # for i in range(len(mosaiclist)):
    #    dfs.append(pd.DataFrame.from_dict({"Image": [i],
    #                                       "real-label": [labels[i]],
    #                                       "predictive-set": [sets[i]],
    #                                       }))
    #    print(f"Image {i} has label \'{labels[i]}\', and the predictive set is {sets[i]}.")
    # write txt
    # df = pd.concat(dfs)
    # df_results_path = os.path.join(
    #    OUTPUT_CP_DIR, f"{_model}", config_file_name, hashed_data, "explore-results_notune.csv")
    # os.makedirs(os.path.dirname(df_results_path), exist_ok=True)
    # df.to_csv(df_results_path)
