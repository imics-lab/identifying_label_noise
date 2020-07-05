#Author: Gentry Atkinson
#Organization: Texas University
#Data: 04 July, 2020
#This file will identify the 10 most suspect indexes in the Sussex-Huawei dataset
#and will write to file

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE as tsne
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics

NUM_OF_RUNS = 5

if __name__ == "__main__":
    data_file = "src/datasets/huawei1_data.csv"
    label_file = "src/datasets/huawei1_labels.csv"
    feature_file = "src/datasets/huawei1_features.csv"

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

    all_bad = all_bad[:10]
    print("Bad indexes in fall data: ", all_bad)
    np.savetxt("huawei_hand_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))

    plt.figure(1)
    plt.scatter(e[:,0], e[:,1], s=2, c=labels)
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='+', s=20, c='red')
    plt.title("Mislabeled Instances in UniMib Fall")
    plt.savefig('UniMib_fall_bad_instances.pdf')

    data_file = "src/datasets/huawei2.csv"
    label_file = "src/datasets/huawei2_labels.csv"
    feature_file = "src/datasets/huawei2_features.csv"

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

    all_bad = all_bad[:10]
    print("Bad indexes in all class data: ", all_bad)
    np.savetxt("huawei_torso_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))

    plt.figure(2)
    plt.scatter(e[:,0], e[:,1], s=2, c=labels)
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='+', s=20, c='red')
    plt.title("Mislabeled Instances in UniMib All Class")
    plt.savefig('UniMib_fall_all_class_instances.pdf')

    plt.show()
