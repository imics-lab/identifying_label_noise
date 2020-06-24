#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 June, 2020
#This code will read the Huawei txt files into CSV and rename them to fit the
#conventions being used in the 3 test files

import numpy as np
import random
import gc
from scipy.io import loadmat
from ts_feature_toolkit import get_features_for_set


NUM_LINES = 196072
SAMPLE_SIZE = 500

if __name__ == "__main__":
    x_filename = "src/datasets/Huawei/Acc_x.txt"
    y_filename = "src/datasets/Huawei/Acc_y.txt"
    z_filename = "src/datasets/Huawei/Acc_z.txt"

    label_file = "src/datasets/Huawei/Acc_labels.csv"

    x_file = open(x_filename, 'r')
    y_file = open(y_filename, 'r')
    z_file = open(z_filename, 'r')

    data = np.zeros((NUM_LINES, SAMPLE_SIZE))
    line = 0

    while(True):
        x_line = x_file.readline()
        y_line = y_file.readline()
        z_line = z_file.readline()
        if (x_line=='' or y_line=='' or z_line==''):
            break

        x_array = x_line.split(" ")
        y_array = y_line.split(" ")
        z_array = z_line.split(" ")

        for j in range(SAMPLE_SIZE):
            data[line][j] = np.linalg.norm([x_array[j], y_array[j], z_array[j]])


        if line % 1000 == 0:
            print(line, ": ", data[line][0])
        line += 1


    print("Data array filled")

    features = get_features_for_set(data, num_samples=len(data))
    gc.collect()

    print("Feature array filled")

    np.savetxt("src/datasets/huawei1_data.csv", data, delimiter=",")
    np.savetxt("src/datasets/huawei1_features.csv", features, delimiter=",")

    print("Data and Features written to CSV")
    data = 0
    features = 0
    gc.collect()

    print("Adding label noise to label array")

    labels_array = np.genfromtxt(label_file, delimiter=' ', dtype='int')
    labels = np.zeros(len(labels_array), dtype='int')
    labels_with_noise = np.zeros(len(labels_array), dtype='int')
    gremlinCounter = 0
    NUM_CLASSES = np.max(labels)
    badIndexes = np.array([], dtype='int')
    for i in range(len(labels)):
        labels[i] = (np.rint(np.mean(labels_array[i])))-1
        gremlin = random.randint(0,100)
        if gremlin < 3:
            labels_with_noise[i] = (labels[i]+1)%NUM_CLASSES
            badIndexes = np.append(badIndexes, [i])
            gremlinCounter += 1
        else:
            labels_with_noise[i] = labels[i]

    np.savetxt("src/datasets/huawei1_labels.csv", labels, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/huawei_with_noise1_labels.csv", labels_with_noise, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/huawei_with_noise1_indexes.csv", badIndexes, delimiter=",", fmt="%d")

    print(gremlinCounter, " bad labels")
    print("All done")
