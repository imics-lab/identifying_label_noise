#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test an
#SVM trained with and without cleaned datagen

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
from utils.gen_ts_data import generate_pattern_data_as_dataframe
import numpy as np
import pandas as pd
import gc
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

    df = pd.DataFrame(data={'id': id, 'time': time, 'x': val}, columns=['id', 'time', 'x'], dtype='int32')
    print(df)
    return df

def cast_dataframe_to_array(X, numSamples):
    length = int(len(X)/numSamples)
    print("This data frame has ", len(X), " samples")
    print("This data frame has ", length, " length")
    print("This data is a ", type(X))
    array = np.zeros((numSamples, length))
    data = np.reshape(X[['x']].to_numpy(), newshape=(length*numSamples))
    for i in range(numSamples):
        array[i, :] = data[i*length:(i+1)*length]
    #print("I made an array from a dataframe:\n", array)
    return array



def get_best_features(X, y):
    ext = extract_features(X, column_id="id", column_sort="time")
    imp = impute(ext)
    sel = select_features(ext, y)
    #print("Features selected from data")
    #print(sel)
    return sel

if __name__ == "__main__":
    print("creating 500 time series sequences with 3 labels over 5 test runs")
    NUM_SAMPLES = 500
    LENGTH = 500
    NUM_OF_RUNS = 3

    raw_precision = np.zeros((NUM_OF_RUNS))
    cleaned_precision = np.zeros((NUM_OF_RUNS))
    classifier = svm.LinearSVC(verbose=0)

    for iter_num in range(NUM_OF_RUNS):
        print("--------------Run Number: ", iter_num+1, "--------------------")
        #generate data in 3 classes
        raw_data, labels = generate_pattern_data_as_dataframe(length=LENGTH, numSamples=NUM_SAMPLES, numClasses=3)

        #extract features and convert everything to arrays
        data_features = get_best_features(raw_data, labels)
        raw_data = cast_dataframe_to_array(raw_data, 500)
        data_features = data_features.to_numpy()
        #print(data_features)

        #pre-process and identify data
        raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)
        #data_features, labels = preprocess_x_y_and_shuffle(data_features, labels)

        #generate list of most poorly fit indexes
        res = check_dataset(raw_data, labels)

        #train and test on raw features
        X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        raw_precision[iter_num] = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
        print_statistics(data_features, labels)

        #remove 2% worst fit samples
        print("Removing top 2%")
        counter = 0
        index_list = np.array(res["indices"][:10])
        index_list = np.sort(index_list)
        print("Indexes to remove: ", index_list)
        #index_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(len(index_list)):
            data_features = np.delete(data_features, index_list[i]-counter, 0)
            labels = np.delete(labels, index_list[i]-counter)
            counter += 1
        assert counter==10, "Wrong number of samples removed"

        #train and test on cleaned data
        X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cleaned_precision[iter_num] = sklearn.metrics.precision_score(y_test, y_pred, average='macro')

        #check for reasonability
        print_statistics(data_features, labels)

        #clean up loose ends in memory
        gc.collect()

    print("\n\n--------Precision of Raw SVM----------------")
    for val in raw_precision:
        print(val)

    print("\n\n--------Precision of Cleaned SVM------------")
    for val in cleaned_precision:
        print(val)
