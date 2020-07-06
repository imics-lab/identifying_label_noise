#Author: Gentry Atkinson
#Organization: Texas University
#Data: 01 July, 2020
#This file will identify the 10 most suspect indexes in the UniMib SHAR dataset
#and will write to file

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE as tsne
from labelfix import check_dataset, preprocess_x_y_and_shuffle
from scipy.io import loadmat

NUM_OF_RUNS = 5

if __name__ == "__main__":
    # data_file = "src/datasets/unimib1_data.csv"
    # label_file = "src/datasets/unimib1_labels.csv"
    # feature_file = "src/datasets/unimib1_features.csv"
    #
    #
    # raw_data = np.genfromtxt(data_file, delimiter=',')
    # labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    # names = ['No Fall', 'Fall']
    # preprocess_x_y_and_shuffle(raw_data, labels)
    #
    # first_run = True
    #
    # # for i in range(NUM_OF_RUNS):
    # #     print("--------------Run Number: ", i+1, "--------------------")
    # #     res_ts = check_dataset(raw_data, labels)
    # #
    # #     bad = np.array(res_ts["indices"][:100])
    # #     if first_run:
    # #         all_bad = bad
    # #         first_run = False
    # #     else:
    # #         all_bad = np.intersect1d(bad, all_bad)
    # #     print(all_bad[:10])
    #
    # all_bad = np.genfromtxt("unimib_fall_bad_indexes.csv", delimiter=',', dtype='int')
    # all_bad = all_bad[:10]
    # print("Bad indexes in fall data: ", all_bad)
    # np.savetxt("unimib_fall_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")
    #
    # #e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))
    # e = np.genfromtxt("src/datasets/unimib1_tsne.csv", delimiter=',')
    #
    # plt.figure(1)
    # plt.scatter(e[labels==0,0], e[labels==0,1], s=2, c='blue', label="No Fall")
    # plt.scatter(e[labels==1,0], e[labels==1,1], s=2, c='green', label="Fall")
    # plt.scatter(e[all_bad,0], e[all_bad,1], marker='+', s=75, c='red', label="Mislabeled")
    # plt.title("UniMib Fall t-SNE visualization")
    # plt.axis('off')
    # plt.legend()
    # plt.savefig('UniMib_fall_bad_instances.pdf')
    #np.savetxt("src/datasets/unimib1_tsne.csv", e, delimiter=",", fmt="%d")

    data_file = "src/datasets/unimib2_data.csv"
    label_file = "src/datasets/unimib2_labels.csv"
    feature_file = "src/datasets/unimib2_features.csv"
    name_file = "src/datasets/UniMiB-SHAR/data/acc_names.mat"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    preprocess_x_y_and_shuffle(raw_data, labels)
    names=['StandFL',
        'Walk',
        'Upstaris',
        'Downstairs',
        'Lie',
        'Sit',
        'FallF',
        'FallR',
        'FallB',
        'Hit',
        'FallProj',
        'FallSit',
        'Syncope',
        'FallL',
        'StandFS',
        'Run',
        'Jump']
    grays = [ '#111111',
        '#222222',
        '#333333',
        '#444444',
        '#555555',
        '#666666',
        '#717171',
        '#767676',
        '#818181',
        '#868686',
        '#919191',
        '#969696',
        '#a1a1a1',
        '#a6a6a6',
        '#b1b1b1',
        '#b6b6b6',
        '#c1c1c1'
    ]

    colors = ['blue',
        'green',
        'orange',
        'cyan',
        'magenta',
        'purple',
        'pink',
        'gray',
        'olive',
        'brown',
        'darkblue',
        'gold',
        'darkgreen',
        'deepskyblue',
        'lawngreen',
        'black',
        'turquoise']

    first_run = True

    # for i in range(NUM_OF_RUNS):
    #     print("--------------Run Number: ", i+1, "--------------------")
    #     res_ts = check_dataset(raw_data, labels)
    #
    #     bad = np.array(res_ts["indices"][:100])
    #     if first_run:
    #         all_bad = bad
    #         first_run = False
    #     else:
    #         all_bad = np.intersect1d(bad, all_bad)
    #     print(all_bad[:10])

    all_bad = np.genfromtxt("unimib_all_class_bad_indexes.csv", delimiter=',', dtype='int')
    all_bad = all_bad[:10]
    print("Bad indexes in all class data: ", all_bad)
    np.savetxt("unimib_all_class_bad_indexes.csv", all_bad, delimiter=",", fmt="%d")

    #e = tsne(n_components=2, n_jobs=8).fit_transform(np.genfromtxt(feature_file, delimiter=','))
    e = np.genfromtxt("src/datasets/unimib2_tsne.csv", delimiter=',')
    #np.savetxt("src/datasets/unimib2_tsne.csv", e, delimiter=",", fmt="%d")

    plt.figure(2)
    print(e[:,0])
    print(e[:,1])
    print("Len e:", len(e))
    print("Len labels: ", len(labels))
    #plt.scatter(e[:,0], e[:,1], s=2, c=labels)
    for i in range(17):
        x = np.where(labels==i)
        plt.scatter(e[x, 0], e[x, 1], c=grays[i], s=4, label=names[i])
    plt.scatter(e[all_bad,0], e[all_bad,1], marker='x', s=200, c='red', label="Mislabeled")
    plt.title("UniMib All Class t-SNE visualization")
    plt.axis('off')
    plt.legend()
    plt.savefig('UniMib_all_class_instances.pdf')

    plt.show()
