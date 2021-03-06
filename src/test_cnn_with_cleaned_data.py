#Author: Gentry Atkinson
#Organization: Texas University
#Data: 18 June, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test a
#CNN trained with and without cleaned data

import sys
import gc
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Reshape, Input
from tensorflow.keras.utils import to_categorical

def decode_from_one_hot(Y):
    retArray = np.zeros(len(Y))
    for i in range(len(Y)):
        retArray[i] = np.argmax(Y[i])
    return retArray

ONLY_CLEAN_TRAIN = False


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

    f = open("data_cleaning_experiments_results.txt", 'a')

    f.write("Running CNN test on data set "+NAME + str(DATASET_NUM)+"\n")
    if ONLY_CLEAN_TRAIN:
        f.write("Only cleaning training data\n")
    else:
        f.write("Cleaning train and test data\n")

    data_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_labels.csv"

    raw_precision = np.zeros((NUM_OF_RUNS))
    cleaned_precision = np.zeros((NUM_OF_RUNS))
    raw_accuracy = np.zeros((NUM_OF_RUNS))
    cleaned_accuracy = np.zeros((NUM_OF_RUNS))
    raw_recall = np.zeros((NUM_OF_RUNS))
    cleaned_recall = np.zeros((NUM_OF_RUNS))

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    NUM_CLASSES = max(labels)+1
    NUM_SAMPLES = len(raw_data)

    raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)
    f.write(str(len(raw_data))+ " samples in dataset\n")
    f.write(str(len(labels))+ " labels in dataset\n")
    f.write(str(NUM_CLASSES)+ " distinct labels\n")

    classifier = Sequential([
        Input(shape=(len(raw_data[0]))),
        Reshape((len(raw_data[0]),1)),
        Conv1D(filters=64, kernel_size=16, activation='relu', padding='same', use_bias=False),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])



    for iter_num in range(NUM_OF_RUNS):
        f.write("--------------Run Number: "+ str(iter_num+1)+ "--------------------\n")

        #train and test on raw features
        X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2, shuffle=True)

        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        classifier.summary()

        if ONLY_CLEAN_TRAIN:
            res_ts = check_dataset(X_train, y_train)
        else:
            res_ts = check_dataset(raw_data, labels)

        y_train = to_categorical(y_train)
        classifier.fit(X_train, y_train, epochs=15, verbose=0)
        y_pred = classifier.predict(X_test)
        y_pred = decode_from_one_hot(y_pred)
        raw_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        raw_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        raw_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

        if ONLY_CLEAN_TRAIN:
            cleaned_data = X_train
            cleaned_labels = y_train
        else:
            cleaned_data = raw_data
            cleaned_labels = labels

        print("Removing top 2% as ts data")
        rem_percent = int(len(cleaned_data) * 0.02)
        index_list = np.array(res_ts["indices"][:rem_percent])
        index_list = np.sort(index_list)
        f.write("Indexes to remove: " + str(index_list) +"\n")
        cleaned_data = np.delete(cleaned_data, index_list, 0)
        cleaned_labels = np.delete(cleaned_labels, index_list, 0)

        #train and test on cleaned data
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        classifier.summary()
        if ONLY_CLEAN_TRAIN:
            classifier.fit(cleaned_data, cleaned_labels, epochs=15, verbose=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(cleaned_data, cleaned_labels, test_size=0.2, shuffle=True)
            y_train = to_categorical(y_train)
            classifier.fit(X_train, y_train, epochs=15, verbose=0)

        y_pred = classifier.predict(X_test)
        y_pred = decode_from_one_hot(y_pred)
        cleaned_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        cleaned_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

        gc.collect()
        f.flush()

    f.write("\n\n--------Results----------------\n")
    for i in range(NUM_OF_RUNS):
        f.write("---Run "+ str(i+1)+ "---\n")
        f.write("Raw precision: "+ str(raw_precision[i])+ "\tRaw accuracy: "+ str(raw_accuracy[i])+ "\tRaw recall: "+ str(raw_recall[i]) + "\n")
        f.write("Cleaned precision: "+ str(cleaned_precision[i])+ "\tCleaned with ts accuracy: "+ str(cleaned_accuracy[i])+ "\tCleaned with ts recall: "+ str(cleaned_recall[i])+"\n")
        f.write("\n")

    f.flush()
    f.close()
