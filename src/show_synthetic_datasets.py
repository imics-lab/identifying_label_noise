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


    X = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    bad_indexes = np.genfromtxt(index_file, delimiter=',', dtype='int')

    print("Creating visualization of synthetic dataset #", DATASET_NUM)

    e = tsne(n_components=2, n_jobs=8).fit_transform(X)

    plt.figure(1)
    plt.scatter(e[:,0], e[:,1], s=2, c=labels)
    plt.title("Features From Set " + str(DATASET_NUM))
    plt.savefig('Synthetic_Set' + str(DATASET_NUM) + '_labels.pdf')

    y = np.zeros(len(labels), dtype='int')
    for i in bad_indexes:
        y[i] = 1

    plt.figure(2)
    cmap = np.array(['g', 'r'])
    plt.scatter(e[:,0], e[:,1], s=6, c=cmap[y])
    plt.title("Mislabeled Points From Set " + str(DATASET_NUM))
    plt.savefig('Synthetic_Set' + str(DATASET_NUM) + '_mislabels.pdf')

    plt.show()
