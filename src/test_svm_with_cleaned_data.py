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
from utils.gen_ts_data import generate_pattern_data_as_dataframe
import numpy as np
import pandas as pd
import random
from tensorflow.keras.utils import to_categorical
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def cast_array_to_dataframe(X):
    numSamples = len(X)
    lengthOfSample = len(X[0])
    id = np.zeros((numSamples*lengthOfSample, 1), dtype='int32')
    time = np.zeros((numSamples*lengthOfSample, 1), dtype='int32')
    val = np.zeros((numSamples*lengthOfSample, 1))
    for i in range(numSamples):
        for j in range(lengthOfSample):
            id[i*lengthOfSample + j] = int(i)
            time[i*lengthOfSample + j] = int(j)
            val[i*lengthOfSample + j] = X[i][j]

    print(type(id))
    print(type(time))
    print(type(val))

    df = pd.DataFrame(data={'id': id, 'time': time, 'x': val}, columns=['id', 'time', 'x'], dtype='int32')
    print(df)
    return df

def cast_dataframe_to_array(X, numSamples):
    length = int(len(X)/numSamples)
    print("This data frame has ", numSamples, " samples")
    print("This data frame has ", length, " length")
    array = np.zeros((numSamples, length))
    data = X['x'].to_numpy()
    for i in range(numSamples):
        array[i][:] = data[i*length:(i+1)*length]
    print("I made an array from a dataframe:\n", array)
    return array



def get_best_features(X, y):
    ext = extract_features(X, column_id="id", column_sort="time")
    imp = impute(ext)
    sel = select_features(ext, y)
    return sel

if __name__ == "__main__":
    print("creating 500 time series sequences with 3 labels")
    NUM_SAMPLES = 500
    LENGTH = 500
    #amplitude = [2, 4, 8]
    #pattern_length = [8, 16, 32]
    #var_pattern_length = [16, 32, 64]
    #var_amplitude = [1, 2, 3]

    #data = np.zeros((NUM_SAMPLES, LENGTH))
    #labels = np.zeros((NUM_SAMPLES, 3))
    #labels = np.zeros(NUM_SAMPLES)

    #generate synthetic data and one hot labels
    #for i in range(NUM_SAMPLES):
    #    label = random.randint(0, 2);
    #    data[i,:]=generate_pattern_data(length=LENGTH, avg_pattern_length=pattern_length[label],
    #        avg_amplitude=amplitude[label], variance_pattern_length=var_pattern_length[label],
    #        variance_amplitude=var_amplitude[label]);
    #    #labels[i, label]=1
    #    labels[i] = label

    raw_data, labels = generate_pattern_data_as_dataframe(length=LENGTH, numSamples=NUM_SAMPLES, numClasses=3)
    data_features = get_best_features(raw_data, labels)
    raw_data = cast_dataframe_to_array(raw_data, 500)
    data_features = cast_dataframe_to_array(data_features, 500)

    #pre-process and identify data
    raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)

    res = check_dataset(raw_data, labels)

    X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
    classifier = svm.LinearSVC(verbose=1)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    prec = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    print("Precision of uncleaned model: ", prec)

    print("Removing top 2%")
    counter = 0
    index_list = np.array(res["indices"][:10])
    index_list = np.sort(index_list)
    print(index_list)
    for i in range(len(index_list)):
        data_features = np.delete(data, i-counter, 0)
        labels = np.delete(labels, i-counter)
        counter += 1

    print("Number of samples removed: ", counter)
    print("New length of data: ", len(data))
    print("New length of labels: ", len(labels))

    X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    prec = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    print("Precision of cleaned model: ", prec)
