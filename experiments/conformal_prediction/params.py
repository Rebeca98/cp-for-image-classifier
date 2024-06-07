from conformal_classification.utils import get_calib_transform , build_model_for_cp
data_path_calibration = 'path/to/calibrationdata'
data_path_test = 'path/to/testdata'
inference_data = 'path/to/inferencedata'
trained_model_path = ''
model_name = ''
num_classes = 20
trained_model = build_model_for_cp(trained_model_path,
                       model_name,
                       num_classes)