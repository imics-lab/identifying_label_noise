#Author: Gentry Atkinson
#Organization: Texas University
#Data: 17 June, 2020
#This code will generate a numpy array of synthetic time series data with known
#bad indexes and then test the precision and recall of removing 1, 2, and 3
#percent of the samples

from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from labelfix import check_dataset, preprocess_x_y_and_shuffle, print_statistics

if __name__ == "__main__":
    NUM_OF_RUNS = 5
    DATASET_NUM = 3

    
