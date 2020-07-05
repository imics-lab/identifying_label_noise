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

    names = ["0", "1", "2", "3", "4"]


    X = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    bad_indexes = np.genfromtxt(index_file, delimiter=',', dtype='int')

    print("Creating visualization of synthetic dataset #", DATASET_NUM)

    e = tsne(n_components=2, n_jobs=8).fit_transform(X)

    plt.figure(1)
    plt.scatter(e[labels==0,0], e[labels==0,1], s=2, c='green', label="first")
    plt.scatter(e[labels==1,0], e[labels==1,1], s=2, c='blue', label="second")
    plt.scatter(e[bad_indexes, 0], e[bad_indexes,1], marker='+', s=60, c='red')
    plt.title("t-SNE Visualization of Synthetic Set " + str(DATASET_NUM))
    plt.axis('off')
    plt.legend()
    plt.savefig('Synthetic_Set' + str(DATASET_NUM) + '.pdf')


    plt.show()
