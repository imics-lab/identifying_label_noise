#Author: Gentry Atkinson
#Organization: Texas University
#Data: 18 June, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test a
#CNN trained with and without cleaned data

import sys
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        NUM_OF_RUNS = 5
        DATASET_NUM = 2

    else:
        NUM_OF_RUNS = int(sys.argv[1])
        DATASET_NUM = int(sys.argv[2])

    f = open("data_cleaning_experiments_results.txt", 'a')

    f.write("Running CNN test on data set: ", DATASET_NUM)

    data_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/synthetic_set"+str(DATASET_NUM)+"_labels.csv"

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
    f.write(len(raw_data), " samples in dataset")
    f.write(len(labels), " labels in dataset")
    f.write(NUM_CLASSES, " distinct labels")

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

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()

    for iter_num in range(NUM_OF_RUNS):
        f.write("--------------Run Number: ", iter_num+1, "--------------------")

        cleaned_data = raw_data
        cleaned_labels = labels

        res_ts = check_dataset(raw_data, labels)

        #train and test on raw features
        X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2, shuffle=False)
        y_train = to_categorical(y_train)
        classifier.fit(X_train, y_train, epochs=7, verbose=0)
        y_pred = classifier.predict(X_test)
        y_pred = decode_from_one_hot(y_pred)
        raw_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        raw_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        raw_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

        print("Removing top 2% as ts data")
        rem_percent = int(NUM_SAMPLES * 0.02)
        index_list = np.array(res_ts["indices"][:rem_percent])
        index_list = np.sort(index_list)
        f.write("Indexes to remove: ", index_list)
        cleaned_data = np.delete(cleaned_data, index_list, 0)
        cleaned_labels = np.delete(cleaned_labels, index_list)

        #train and test on cleaned data
        X_train, X_test, y_train, y_test = train_test_split(cleaned_data, cleaned_labels, test_size=0.2, shuffle=False)
        y_train = to_categorical(y_train)
        classifier.fit(X_train, y_train, epochs=7, verbose=0)
        y_pred = classifier.predict(X_test)
        y_pred = decode_from_one_hot(y_pred)
        cleaned_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        cleaned_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

    f.write("\n\n--------Results----------------")
    for i in range(NUM_OF_RUNS):
        f.write("---Run ", i+1, "---")
        f.write("Raw precision: ", raw_precision[i], "\tRaw accuracy: ", raw_accuracy[i], "\tRaw recall: ", raw_recall[i])
        f.write("Cleaned precision: ", cleaned_precision[i], "\tCleaned with ts accuracy: ", cleaned_accuracy[i], "\tCleaned with ts recall: ", cleaned_recall[i])
        f.write("\n")
