#Author: Gentry Atkinson
#Organization: Texas University
#Data: 04 July, 2020
#This file will identify the 10 most suspect indexes in the Sussex-Huawei dataset
#and will write to file

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE as tsne
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
import gc

NUM_OF_RUNS = 3

if __name__ == "__main__":
    data_file = "src/datasets/huawei1_data.csv"
    label_file = "src/datasets/huawei1_labels.csv"
    feature_file = "src/datasets/huawei1_features.csv"
    names = ["Still", "Walk", "Run", "Bike", "Bus", "Car", "Train", "Subway"]
    grays = [ '#111111',
        '#222222',
        '#555555',
        '#717171',
        '#818181',
        '#919191',
        '#a1a1a1',,
        '#c1c1c1'
    ]

    colors = ['blue',
        'green',
        'cyan',
        'gray',
        'olive',
        'brown',
        'gold',
        'darkgreen']

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    preprocess_x_y_and_shuffle(raw_data, labels)

    first_run = True

    for i in range(NUM_OF_RUNS):
        print("--------------Run Number: ", i+1, "--------------------")
        res_ts = check_dataset(raw_data, labels)

        bad = np.array(res_ts["indices"][:1000])
        if first_run:
            all_bad = bad
            first_run = False
        else:
            all_bad = np.intersect1d(bad, all_bad)
        print(all_bad)
        gc.collect()

    all_bad = all_bad[:10]
    print("Bad indexes in HuaWei hand data: ", all_bad)
    np.savetxt("huawei_hand_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))
    np.savetxt("src/datasets/huawei1_tsne.csv", e, delimiter=",", fmt="%d")

    plt.figure(1)
    for i in range(8):
        x = np.where(labels==i)
        plt.scatter(e[x, 0], e[x, 1], c=grays[i], s=4, label=names[i])
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='x', s=200, c='red', label="Mislabeled")
    plt.title("Mislabeled Instances in Sussex-HuaWei Hand")
    plt.savefig('huawei_hand_bad_instances.pdf')
    gc.collect()

    data_file = "src/datasets/huawei2_data.csv"
    label_file = "src/datasets/huawei2_labels.csv"
    feature_file = "src/datasets/huawei2_features.csv"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    preprocess_x_y_and_shuffle(raw_data, labels)

    first_run = True

    for i in range(NUM_OF_RUNS):
        print("--------------Run Number: ", i+1, "--------------------")
        res_ts = check_dataset(raw_data, labels)

        bad = np.array(res_ts["indices"][:1000])
        if first_run:
            all_bad = bad
            first_run = False
        else:
            all_bad = np.intersect1d(bad, all_bad)
        print(all_bad)
        gc.collect()

    all_bad = all_bad[:10]
    print("Bad indexes in huawei torso data: ", all_bad)
    np.savetxt("huawei_torso_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))
    np.savetxt("src/datasets/huawei2_tsne.csv", e, delimiter=",", fmt="%d")

    plt.figure(2)
    for i in range(8):
        x = np.where(labels==i)
        plt.scatter(e[x, 0], e[x, 1], c=grays[i], s=4, label=names[i])
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='x', s=200, c='red', label="Mislabeled")
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='+', s=75, c='red', label="Mislabeled")
    plt.title("Mislabeled Instances in Sussex-HuaWei Torso")
    plt.savefig('huaWei_torso_bad_instances.pdf')
    gc.collect

    plt.show()
