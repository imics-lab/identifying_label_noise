#Author: Gentry Atkinson
#Organization: Texas University
#Data: 19 June, 2020
#This code will produce 2d plots of one of the synthetic datasets

import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE as tsne

if __name__ == "__main__":
    if len(sys.argv) == 1:
        DATASET_NUM = 1
    else:
        DATASET_NUM = int(sys.argv[1])

    data_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_features.csv"
    label_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_labels.csv"
    index_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_indexes.csv"
    grays = [ '#111111',
        '#555555',
        '#818181',
        '#a1a1a1',
        '#c1c1c1'
    ]

    names = ["0", "1", "2", "3", "4"]


    X = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    bad_indexes = np.genfromtxt(index_file, delimiter=',', dtype='int')

    print("Creating visualization of synthetic dataset #", DATASET_NUM)

    e = tsne(n_components=2, n_jobs=8).fit_transform(X)

    plt.figure(1)
    for i in range(max(labels)+1):
        x = np.where(labels==i)
        plt.scatter(e[x, 0], e[x, 1], c=grays[i], s=4, label=names[i])
    plt.scatter(e[bad_indexes, 0], e[bad_indexes,1], marker='x', s=75, c='red')
    plt.title("Synthetic Set " + str(DATASET_NUM) + " t-SNE Visualization")
    plt.axis('off')
    plt.legend()
    plt.savefig('Synthetic_Set' + str(DATASET_NUM) + '.pdf')


    plt.show()
