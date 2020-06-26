#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test an
#SVM trained with and without cleaned data

import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import normalize
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
import sys

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
    return array



def get_best_features(X, y):
    labels = pd.DataFrame({'y':y})
    sel = extract_relevant_features(X, labels['y'], column_id='id', column_sort='time')
    return sel

if __name__ == "__main__":
    if len(sys.argv) < 3:
        NUM_OF_RUNS = 5
        DATASET_NUM = 2
        NAME = "synthetic_set"
    elif len(sys.argv) < 4:
        NAME = "synthetic_set"
    else:
        NUM_OF_RUNS = int(sys.argv[1])
        DATASET_NUM = int(sys.argv[2])
        NAME = sys.argv[3]

    raw_precision = np.zeros((NUM_OF_RUNS))
    cleaned_precision_as_ts = np.zeros((NUM_OF_RUNS))
    cleaned_precision_as_numerical = np.zeros((NUM_OF_RUNS))
    raw_accuracy = np.zeros((NUM_OF_RUNS))
    cleaned_accuracy_as_ts = np.zeros((NUM_OF_RUNS))
    cleaned_accuracy_as_numerical = np.zeros((NUM_OF_RUNS))
    raw_recall = np.zeros((NUM_OF_RUNS))
    cleaned_recall_as_ts = np.zeros((NUM_OF_RUNS))
    cleaned_recall_as_numerical = np.zeros((NUM_OF_RUNS))

    classifier = svm.LinearSVC(verbose=0, dual=False)

    f = open("data_cleaning_experiments_results.txt", 'a')

    f.write("Running SVM test on data set: " + NAME + str(DATASET_NUM) + "\n")
    #read one of three data sets with 3 classes
    data_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_labels.csv"
    feature_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_features.csv"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    f.write(str(len(raw_data)) + " samples in dataset\n")
    f.write(str(len(labels)) + " labels in dataset\n")
    f.write(str(max(labels)+1) + " distinct labels\n")
    NUM_SAMPLES = len(raw_data)
    #extract features
    #data_features = get_features_for_set(raw_data, num_samples=NUM_SAMPLES)
    data_features = np.genfromtxt(feature_file, delimiter=',')
    normalize(data_features, copy='False', axis=0)

    #pre-process and identify data
    raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)




    for iter_num in range(NUM_OF_RUNS):
        f.write("--------------Run Number: " + str(iter_num+1) + "--------------------\n")

        print("Classes represented in ts data: ", res_ts["Classes"])
        print("Classes represented in numerical data: ", res_numercical["Classes"])

        #train and test on raw features
        X_raw, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2, shuffle=True)
        X_train = get_features_for_set(X_raw, len(X_raw))
        cleaned_features = X_train
        cleaned_labels = y_train

        #generate list of most poorly fit indexes
        res_ts = check_dataset(X_raw, y_train)
        res_numercical = check_dataset(X_train, y_train, hyperparams={
            "input_dim": data_features.shape[1],
            "output_dim": max(labels)+1,
            "num_hidden": 3,
            "size_hidden": 50,
            "dropout": 0.1,
            "epochs": 400,
            "learn_rate": 1e-2,
            "activation": "relu"
        })


        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        raw_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        raw_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        raw_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

        #check for reasonability
        print_statistics(data_features, labels)

        #remove 2% worst fit samples using ts model
        print("Removing top 2% as ts data")
        rem_percent = int(NUM_SAMPLES * 0.02)
        index_list = np.array(res_ts["indices"][:rem_percent])
        index_list = np.sort(index_list)
        f.write("Indexes to remove from ts analysis: " +  str(index_list)+"\n")
        cleaned_features = np.delete(cleaned_features, index_list, 0)
        cleaned_labels = np.delete(cleaned_labels, index_list)

        #train and test on cleaned data
        #X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.2, shuffle=False)
        classifier.fit(cleaned_features, cleaned_labels)
        y_pred = classifier.predict(X_test)
        cleaned_precision_as_ts[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy_as_ts[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        cleaned_recall_as_ts[iter_num] = recall_score(y_test, y_pred, average='macro')

        #check for reasonability
        print_statistics(data_features, labels)

        cleaned_features = X_train
        cleaned_labels = y_train

        #remove 2% worst fit samples using numerical model
        print("Removing top 2% as numerical data")
        rem_percent = int(NUM_SAMPLES * 0.02)
        index_list = np.array(res_numercical["indices"][:rem_percent])
        index_list = np.sort(index_list)
        f.write("Indexes to remove from numerical analysis: " + str(index_list)+"\n")
        cleaned_features = np.delete(cleaned_features, index_list, 0)
        cleaned_labels = np.delete(cleaned_labels, index_list)

        #train and test on cleaned data
        #X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.2, shuffle=False)
        classifier.fit(cleaned_features, cleaned_labels)
        y_pred = classifier.predict(X_test)
        cleaned_precision_as_numerical[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy_as_numerical[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        cleaned_recall_as_numerical[iter_num] = recall_score(y_test, y_pred, average='macro')

        #check for reasonability
        print_statistics(data_features, labels)

        #clean up loose ends in memory
        gc.collect()

    f.write("\n\n--------Results----------------\n")
    for i in range(NUM_OF_RUNS):
        f.write("---Run " + str(i+1) + "---\n")
        f.write("Raw precision: " + str(raw_precision[i]) + "\tRaw accuracy: " + str(raw_accuracy[i]) + "\tRaw recall: " + str(raw_recall[i])+"\n")
        f.write("Cleaned with ts precision: " + str(cleaned_precision_as_ts[i]) + "\tCleaned with ts accuracy: " + str(cleaned_accuracy_as_ts[i]) + "\tCleaned with ts recall: " + str(cleaned_recall_as_ts[i])+"\n")
        f.write("Cleaned with numerical precision: " + str(cleaned_precision_as_numerical[i]) + "\tCleaned with numerical accuracy: " + str(cleaned_accuracy_as_numerical[i]) + "\tCleaned with numerical recall: " + str(cleaned_recall_as_numerical[i])+"\n")
        f.write("\n")

    f.flush()
    f.close()
