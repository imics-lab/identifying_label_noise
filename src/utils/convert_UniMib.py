#Author: Gentry Atkinson
#Organization: Texas University
#Data: 17 June, 2020
#This code will read the UniMib mat files into CSV and rename them to fit the
#conventions being used in the 3 test files

import numpy as np
import random
from scipy.io import loadmat
from ts_feature_toolkit import get_features_for_set


if __name__ == "__main__":
    twoClass_X = loadmat('src/datasets/UniMiB-SHAR/data/two_classes_data.mat')['two_classes_data']
    twoClass_y = loadmat('src/datasets/UniMiB-SHAR/data/two_classes_labels.mat')['two_classes_labels'][:,0]
    twoClass_feat = get_features_for_set(twoClass_X, num_samples=len(twoClass_X))

    twoClass_y = twoClass_y - 1

    np.savetxt("src/datasets/unimib1_data.csv", twoClass_X, delimiter=",")
    np.savetxt("src/datasets/unimib1_labels.csv", twoClass_y, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/unimib1_features.csv", twoClass_feat, delimiter=",")

    gremlinCounter = 0
    badIndexes = np.array([], dtype='int')
    for i in range(len(twoClass_y)):
        gremlin = random.randint(0,100)
        if gremlin < 3:
            twoClass_y[i] = (twoClass_y[i]+1)%2
            badIndexes = np.append(badIndexes, [i])
            gremlinCounter += 1
    print (gremlinCounter, " bad indexes in two class data")

    np.savetxt("src/datasets/unimib_with_noise1_labels.csv", twoClass_y, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/unimib_with_noise1_features.csv", twoClass_feat, delimiter=",")
    np.savetxt("src/datasets/unimib_with_noise1_data.csv", twoClass_X, delimiter=",")
    np.savetxt("src/datasets/unimib_with_noise1_indexes.csv", badIndexes, delimiter=",", fmt="%d")

    allClass_X = loadmat('src/datasets/UniMiB-SHAR/data/acc_data.mat')['acc_data']
    allClass_y = loadmat('src/datasets/UniMiB-SHAR/data/acc_labels.mat')['acc_labels'][:,0]
    allClass_feat = get_features_for_set(allClass_X, num_samples=len(allClass_X))

    allClass_y = allClass_y - 1

    np.savetxt("src/datasets/unimib2_data.csv", allClass_X, delimiter=",")
    np.savetxt("src/datasets/unimib2_labels.csv", allClass_y, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/unimib2_features.csv", allClass_feat, delimiter=",")

    gremlinCounter = 0
    badIndexes = np.array([], dtype='int')
    for i in range(len(allClass_y)):
        gremlin = random.randint(0,100)
        if gremlin < 3:
            allClass_y[i] = (allClass_y[i]+1)%2
            badIndexes = np.append(badIndexes, [i])
            gremlinCounter += 1
    print (gremlinCounter, " bad indexes in all class data")
    np.savetxt("src/datasets/unimib_with_noise2_data.csv", allClass_X, delimiter=",")
    np.savetxt("src/datasets/unimib_with_noise2_labels.csv", allClass_y, delimiter=",", fmt="%d")
    np.savetxt("src/datasets/unimib_with_noise2_features.csv", allClass_feat, delimiter=",")
    np.savetxt("src/datasets/unimib_with_noise2_indexes.csv", badIndexes, delimiter=",", fmt="%d")
