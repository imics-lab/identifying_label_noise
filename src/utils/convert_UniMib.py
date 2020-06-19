#Author: Gentry Atkinson
#Organization: Texas University
#Data: 17 June, 2020
#This code will read the UniMib mat files into CSV and rename them to fit the
#conventions being used in the 3 test files

import numpy as np
from scipy.io import loadmat
from ts_feature_toolkit import get_features_for_set


if __name__ == "__main__":
    twoClass_X = loadmat('src/datasets/UniMiB-SHAR/data/two_classes_data.mat')['two_classes_data']
    twoClass_y = loadmat('src/datasets/UniMiB-SHAR/data/two_classes_labels.mat')['two_classes_labels'][:,0]
    twoClass_feat = get_features_for_set(twoClass_X, num_samples=len(twoClass_X))

    np.savetxt("src/datasets/unimib1_data.csv", twoClass_X, delimiter=",")
    np.savetxt("src/datasets/unimib1_labels.csv", twoClass_y, delimiter=",")
    np.savetxt("src/datasets/unimib1_features.csv", twoClass_feat, delimiter=",")
