#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas University
#Data: 18 June, 2020
#This script will generate 4 synthetic data sets and run 12 experiments to
#measure the effectiveness of extended labelfix on timeseries data.

#python3 src/create_ts_data.py

python3 src/test_data_cleaning_for_precision.py 5 1
python3 src/test_data_cleaning_for_precision.py 5 2
python3 src/test_data_cleaning_for_precision.py 5 3
python3 src/test_data_cleaning_for_precision.py 5 4

python3 src/test_svm_with_cleaned_data.py 5 1
python3 src/test_svm_with_cleaned_data.py 5 2
python3 src/test_svm_with_cleaned_data.py 5 3
python3 src/test_svm_with_cleaned_data.py 5 4

python3 src/test_cnn_with_cleaned_data.py 5 1
python3 src/test_cnn_with_cleaned_data.py 5 2
python3 src/test_cnn_with_cleaned_data.py 5 3
python3 src/test_cnn_with_cleaned_data.py 5 4
