#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas University
#Data: 18 June, 2020
#This script will generate 4 synthetic data sets and run 12 experiments to
#measure the effectiveness of extended labelfix on timeseries data.

#python3 src/create_ts_data.py
#python3 src/utils/convert_Huawei.py
#python3 src/utils/convert_UniMib.py

# python3 src/test_svm_with_cleaned_data.py 5 1 synthetic_set
# python3 src/test_svm_with_cleaned_data.py 5 2 synthetic_set
# python3 src/test_svm_with_cleaned_data.py 5 3 synthetic_set
# python3 src/test_svm_with_cleaned_data.py 5 4 synthetic_set
# python3 src/test_svm_with_cleaned_data.py 5 1 huawei
# python3 src/test_svm_with_cleaned_data.py 5 2 huawei
# python3 src/test_svm_with_cleaned_data.py 5 1 unimib
# python3 src/test_svm_with_cleaned_data.py 5 2 unimib
#
# python3 src/test_cnn_with_cleaned_data.py 5 1 synthetic_set
# python3 src/test_cnn_with_cleaned_data.py 5 2 synthetic_set
# python3 src/test_cnn_with_cleaned_data.py 5 3 synthetic_set
# python3 src/test_cnn_with_cleaned_data.py 5 4 synthetic_set
# python3 src/test_cnn_with_cleaned_data.py 5 1 huawei
# python3 src/test_cnn_with_cleaned_data.py 5 2 huawei
python3 src/test_cnn_with_cleaned_data.py 5 1 unimib
python3 src/test_cnn_with_cleaned_data.py 5 2 unimib

# python3 src/test_data_cleaning_for_precision.py 5 1 synthetic_set
# python3 src/test_data_cleaning_for_precision.py 5 2 synthetic_set
# python3 src/test_data_cleaning_for_precision.py 5 3 synthetic_set
# python3 src/test_data_cleaning_for_precision.py 5 4 synthetic_set
# python3 src/test_data_cleaning_for_precision.py 5 1 huawei_with_noise
# python3 src/test_data_cleaning_for_precision.py 5 2 huawei_with_noise
# python3 src/test_data_cleaning_for_precision.py 5 1 unimib_with_noise
# python3 src/test_data_cleaning_for_precision.py 5 2 unimib_with_noise
