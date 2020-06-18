#Author: Gentry Atkinson
#Organization: Texas University
#Data: 18 June, 2020
#This code will generate a numpy array of synthetic time series data, use the
#labelfix preprocessor to flip mu percent (0.03) of the labels, and then test a
#CNN trained with and without cleaned data

import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Reshape, Input

if __name__ == "__main__":
    if len(sys.argv) < 3:
        NUM_OF_RUNS = 5
        DATASET_NUM = 2

    else:
        NUM_OF_RUNS = int(sys.argv[1])
        DATASET_NUM = int(sys.argv[2])

    print("Running CNN test on data set: ", DATASET_NUM)

    data_file = "src/datasets/accuracy_test"+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/accuracy_test"+str(DATASET_NUM)+"_labels.csv"

    raw_precision = np.zeros((NUM_OF_RUNS))
    cleaned_precision_as = np.zeros((NUM_OF_RUNS))
    raw_accuracy = np.zeros((NUM_OF_RUNS))
    cleaned_accuracy_as = np.zeros((NUM_OF_RUNS))
    raw_recall = np.zeros((NUM_OF_RUNS))
    cleaned_recall_as = np.zeros((NUM_OF_RUNS))

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    print(len(raw_data), " samples in dataset")
    print(len(labels), " labels in dataset")
    print(max(labels)+1, " distinct labels")
    NUM_SAMPLES = len(raw_data)

    raw_data, labels = preprocess_x_y_and_shuffle(raw_data, labels)

    classifier = Sequential([
        Input(shape=shape_x),
        Reshape((shape_x,1)),
        Conv1D(filters=64, kernel_size=16, activation='relu', padding='same', use_bias=False),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(shape_y, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for iter_num in range(NUM_OF_RUNS):
        print("--------------Run Number: ", iter_num+1, "--------------------")

        cleaned_data = raw_data
        cleaned_labels = labels

        res_ts = check_dataset(raw_data, labels)

        #train and test on raw features
        X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train, epochs=10, )
        y_pred = classifier.predict(X_test)
        raw_precision[iter_num] = precision_score(y_test, y_pred, average='macro')
        raw_accuracy[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        raw_recall[iter_num] = recall_score(y_test, y_pred, average='macro')

        print("Removing top 2% as ts data")
        rem_percent = int(NUM_SAMPLES * 0.02)
        index_list = np.array(res_ts["indices"][:rem_percent])
        index_list = np.sort(index_list)
        print("Indexes to remove: ", index_list)
        cleaned_data = np.delete(cleaned_data, index_list, 0)
        cleaned_labels = np.delete(cleaned_labels, index_list)

        #train and test on cleaned data
        X_train, X_test, y_train, y_test = train_test_split(cleaned_data, cleaned_labels, test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cleaned_precision_as_ts[iter_num] = precision_score(y_test, y_pred, average='macro')
        cleaned_accuracy_as_ts[iter_num] = accuracy_score(y_test, y_pred, normalize=True)
        cleaned_recall_as_ts[iter_num] = recall_score(y_test, y_pred, average='macro')

    print("\n\n--------Results----------------")
    for i in range(NUM_OF_RUNS):
        print("---Run ", i+1, "---")
        print("Raw precision: ", raw_precision[i], "\tRaw accuracy: ", raw_accuracy[i], "\tRaw recall: ", raw_recall[i])
        print("Cleaned with ts precision: ", cleaned_precision_as_ts[i], "\tCleaned with ts accuracy: ", cleaned_accuracy_as_ts[i], "\tCleaned with ts recall: ", cleaned_recall_as_ts[i])
        print("Cleaned with numerical precision: ", cleaned_precision_as_numerical[i], "\tCleaned with numerical accuracy: ", cleaned_accuracy_as_numerical[i], "\tCleaned with numerical recall: ", cleaned_recall_as_numerical[i])
        print("\n")
