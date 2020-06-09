#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test an
#SVM trained with and without cleaned datagen

import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
from utils.gen_ts_data import generate_pattern_data_as_dataframe
from utils.ts_feature_toolkit import get_features_for_set
import numpy as np
import pandas as pd
import gc
from tensorflow.keras.utils import to_categorical
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import matplotlib.pyplot as plt

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
    labels = pd.DataFrame({'y':y})
    #ext = extract_features(X, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters())
    #imp = impute(ext)
    #sel = select_features(imp, y,  fdr_level=0.02, ml_task='classification')
    sel = extract_relevant_features(X, labels['y'], column_id='id', column_sort='time')
    return sel

if __name__ == "__main__":
    #print("creating 500 time series sequences with 3 labels over 5 test runs")
    NUM_OF_RUNS = 5
    DATASET_NUM = 1

    raw_precision = np.zeros((NUM_OF_RUNS))
    cleaned_precision = np.zeros((NUM_OF_RUNS))
    raw_accuracy = np.zeros((NUM_OF_RUNS))
    cleaned_accuracy = np.zeros((NUM_OF_RUNS))
    classifier = svm.LinearSVC(verbose=0)

    print("Running test on data set: ", DATASET_NUM)
    #read one of three data sets with 3 classes
    data_file = "src/datasets/svm_test"+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/svm_test"+str(DATASET_NUM)+"_labels.csv"


    for iter_num in range(NUM_OF_RUNS):
        print("--------------Run Number: ", iter_num+1, "--------------------")

        raw_data = np.genfromtxt(data_file, delimiter=',')
        labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
        print(len(raw_data), " samples in dataset")
        print(len(labels), " labels in dataset")
        print(max(labels)+1, " distinct labels")
        NUM_SAMPLES = len(raw_data)
        #extract features
        data_features = get_features_for_set(raw_data, num_samples=NUM_SAMPLES)

        #pre-process and identify data
        raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)

        #generate list of most poorly fit indexes
        res = check_dataset(raw_data, labels)
        print("Classes represented in this data: ", res["Classes"])

        #train and test on raw features
        X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        raw_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        raw_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)

        #check for reasonability
        print_statistics(data_features, labels)

        #remove 2% worst fit samples
        print("Removing top 2%")
        counter = 0
        rem_percent = int(NUM_SAMPLES * 0.02)
        index_list = np.array(res["indices"][:rem_percent])
        index_list = np.sort(index_list)
        print("Indexes to remove: ", index_list)
        data_features = np.delete(data_features, index_list, 0)
        labels = np.delete(labels, index_list)

        #train and test on cleaned data
        X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cleaned_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)

        #check for reasonability
        print_statistics(data_features, labels)

        #clean up loose ends in memory
        gc.collect()

    print("\n\n--------Results----------------")
    for i in range(NUM_OF_RUNS):
        print("---Run ", i+1, "---")
        print("Raw precision: ", raw_precision[i], "\tRaw accuracy: ", raw_accuracy[i])
        print("Cleaned precision: ", cleaned_precision[i], "\tCleaned accuracy: ", cleaned_accuracy[i])
        print("\n")

    plt.plot(raw_data[0,:])
    plt.show()
