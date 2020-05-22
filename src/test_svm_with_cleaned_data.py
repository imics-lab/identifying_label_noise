#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test an
#SVM trained with and without cleaned datagen

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from labelfix import check_dataset, preprocess_x_y_and_shuffle
from utils.gen_ts_data import generate_pattern_data
import numpy as np
import random
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    print("creating 500 time series sequences with 3 labels")
    NUM_SAMPLES = 500
    LENGTH = 500
    amplitude = [2, 4, 8]
    pattern_length = [8, 16, 32]
    var_pattern_length = [16, 32, 64]
    var_amplitude = [1, 2, 3]

    data = np.zeros((NUM_SAMPLES, LENGTH))
    #labels = np.zeros((NUM_SAMPLES, 3))
    labels = np.zeros(NUM_SAMPLES)

    #generate synthetic data and one hot labels
    for i in range(NUM_SAMPLES):
        label = random.randint(0, 2);
        data[i,:]=generate_pattern_data(length=LENGTH, avg_pattern_length=pattern_length[label],
            avg_amplitude=amplitude[label], variance_pattern_length=var_pattern_length[label],
            variance_amplitude=var_amplitude[label]);
        #labels[i, label]=1
        labels[i] = label

    #pre-process and identify data
    data, labels = preprocess_x_y_and_shuffle(data, labels)

    res = check_dataset(data, labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=False)
    classifier = svm.LinearSVC()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    prec = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    print("Precision of uncleaned model: ", prec)

    print("Removing top 2%")
    counter = 0
    for i in res["indices"][:10]:
        data = np.delete(data, i-counter, 0)
        labels = np.delete(labels, i-counter)
        counter += 1

    print("Number of samples removed: ", counter)
    print("New length of data: ", len(data))
    print("New length of labels: ", len(labels))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=False)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    prec = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    print("Precision of cleaned model: ", prec)