#Author: Gentry Atkinson
#Organization: Texas University
#Data: 17 June, 2020
#This code will generate a numpy array of synthetic time series data with known
#bad indexes and then test the precision and recall of removing 1, 2, and 3
#percent of the samples

from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics
import sys

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

    ts_precision = np.zeros((NUM_OF_RUNS, 3))
    ts_recall = np.zeros((NUM_OF_RUNS, 3))
    num_precision = np.zeros((NUM_OF_RUNS, 3))
    num_recall = np.zeros((NUM_OF_RUNS, 3))

    f.write("Running precision/recall test on data set: "+ str(DATASET_NUM)+"\n")

    data_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_data.csv"
    label_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_labels.csv"
    feature_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_features.csv"
    index_file = "src/datasets/"+NAME+str(DATASET_NUM)+"_indexes.csv"

    raw_data = np.genfromtxt(data_file, delimiter=',')
    labels = np.genfromtxt(label_file, delimiter=',', dtype='int')
    NUM_SAMPLES = len(raw_data)
    NUM_CLASSES = max(labels)+1
    data_features = np.genfromtxt(feature_file, delimiter=',')
    bad_indexes = np.genfromtxt(index_file, delimiter=',', dtype='int')

    y_true = np.zeros(NUM_SAMPLES)
    for index in bad_indexes :
        y_true[index] = 1

    f.write(str(len(raw_data))+ " samples in dataset\n")
    f.write(str(len(labels))+ " labels in dataset\n")
    f.write(str(NUM_CLASSES)+ " distinct labels\n")

    preprocess_x_y_and_shuffle(raw_data, labels)

    for i in range(NUM_OF_RUNS):
        print("--------------Run Number: ", i+1, "--------------------")
        res_ts = check_dataset(raw_data, labels)
        res_numerical = check_dataset(data_features, labels, hyperparams={
            "input_dim": data_features.shape[1],
            "output_dim": max(labels)+1,
            "num_hidden": 3,
            "size_hidden": 100,
            "dropout": 0.1,
            "epochs": 400,
            "learn_rate": 1e-2,
            "activation": "relu"
        })

        rem_percent = int(NUM_SAMPLES * 0.01)
        ts_y = np.array(res_ts["indices"][:rem_percent])
        num_y = np.array(res_numerical["indices"][:rem_percent])
        y_pred_ts = np.zeros(NUM_SAMPLES)
        for index in ts_y:
            y_pred_ts[index] = 1
        y_pred_num = np.zeros(NUM_SAMPLES)
        for index in num_y:
            y_pred_num[index] = 1
        ts_precision[i, 0] = precision_score(y_true, y_pred_ts, average='macro')
        ts_recall[i, 0] = recall_score(y_true, y_pred_ts, average='macro')
        num_precision[i, 0] = precision_score(y_true, y_pred_num, average='macro')
        num_recall[i, 0] = recall_score(y_true, y_pred_num, average='macro')

        rem_percent = int(NUM_SAMPLES * 0.02)
        ts_y = np.array(res_ts["indices"][:rem_percent])
        num_y = np.array(res_numerical["indices"][:rem_percent])
        y_pred_ts = np.zeros(NUM_SAMPLES)
        for index in ts_y:
            y_pred_ts[index] = 1
        y_pred_num = np.zeros(NUM_SAMPLES)
        for index in num_y:
            y_pred_num[index] = 1
        ts_precision[i, 1] = precision_score(y_true, y_pred_ts, average='macro')
        ts_recall[i, 1] = recall_score(y_true, y_pred_ts, average='macro')
        num_precision[i, 1] = precision_score(y_true, y_pred_num, average='macro')
        num_recall[i, 1] = recall_score(y_true, y_pred_num, average='macro')

        rem_percent = int(NUM_SAMPLES * 0.03)
        ts_y = np.array(res_ts["indices"][:rem_percent])
        num_y = np.array(res_numerical["indices"][:rem_percent])
        y_pred_ts = np.zeros(NUM_SAMPLES)
        for index in ts_y:
            y_pred_ts[index] = 1
        y_pred_num = np.zeros(NUM_SAMPLES)
        for index in num_y:
            y_pred_num[index] = 1
        ts_precision[i, 2] = precision_score(y_true, y_pred_ts, average='macro')
        ts_recall[i, 2] = recall_score(y_true, y_pred_ts, average='macro')
        num_precision[i, 2] = precision_score(y_true, y_pred_num, average='macro')
        num_recall[i, 2] = recall_score(y_true, y_pred_num, average='macro')

    f.write("### Results on Raw Data###\n")
    f.write("\t\tPrec\t\t\tRecall\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.01\t"+ str(ts_precision[i, 0])+ "\t"+ str(ts_recall[i, 0])+"\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.02\t"+ str(ts_precision[i, 1])+ "\t"+ str(ts_recall[i, 1])+"\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.03\t"+ str(ts_precision[i, 2])+ "\t"+ str(ts_recall[i, 2])+"\n")

    f.write("\n\n### Results on Features###")
    f.write("\t\tPrec\t\t\tRecall\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.01\t"+ str(num_precision[i, 0])+ "\t"+ str(num_recall[i, 0])+"\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.02\t"+ str(num_precision[i, 1])+ "\t"+ str(num_recall[i, 1])+"\n")
    for i in range(NUM_OF_RUNS):
        f.write("0.03\t"+ str(num_precision[i, 2])+ "\t"+ str(num_recall[i, 2])+"\n")

    f.flush()
    f.close()
