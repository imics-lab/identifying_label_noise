#Author: Gentry Atkinson
#Organization: Texas University
#Data: 01 July, 2020
#This file will identify the 10 most suspect indexes in the UniMib SHAR dataset
#and will write a visualization of those points to file

import numpy as np
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics

NUM_OF_RUNS = 1

if __name__ == "__main__":
    data_file = "src/datasets/unimib1_data.csv"
    label_file = "src/datasets/unimib1_labels.csv"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    preprocess_x_y_and_shuffle(raw_data, labels)

    first_run = True

    for i in range(NUM_OF_RUNS):
        print("--------------Run Number: ", i+1, "--------------------")
        res_ts = check_dataset(raw_data, labels)

        bad = np.array(res_ts["indices"][:])
        if first_run:
            all_bad = bad
            first_run = False
        else:
            all_bad = np.intersect1d(bad, all_bad)

    print("Bad indexes in fall data: ", all_bad[:10])
    np.savetxt("unimib_fall_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    data_file = "src/datasets/unimib2_data.csv"
    label_file = "src/datasets/unimib2_labels.csv"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    preprocess_x_y_and_shuffle(raw_data, labels)

    first_run = True

    for i in range(NUM_OF_RUNS):
        print("--------------Run Number: ", i+1, "--------------------")
        res_ts = check_dataset(raw_data, labels)

        bad = np.array(res_ts["indices"][:])
        if first_run:
            all_bad = bad
            first_run = False
        else:
            all_bad = np.intersect1d(bad, all_bad)

    print("Bad indexes in all class data: ", all_bad[:10])
    np.savetxt("unimib_fall_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")
