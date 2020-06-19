#Author: Gentry Atkinson
#Organization: Texas University
#Data: 19 June, 2020
#This code will produce 2d plots of one of the synthetic datasets

import sys
from matplotlib import pyplot
from sklearn.manifold import TSNE as tsne

if __name__ == "__main__":
    if len(sys.argv) == 1:
        DATASET_NUM = 1
    else:
        DATASET_NUM = int(sys.argv[1])

    data_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_labels.csv"
