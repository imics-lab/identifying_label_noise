import pandas as pd
import numpy as np
from utils.gen_ts_data import generate_pattern_array_as_csv, generate_pattern_array_as_csv_with_indexes

if __name__ == "__main__":
    generate_pattern_array_as_csv_with_indexes(length=500, numSamples=1000, numClasses=2, percentError=3, filename='src/datasets/synthetic_set1')
    generate_pattern_array_as_csv_with_indexes(length=500, numSamples=1000, numClasses=5, percentError=3, filename='src/datasets/synthetic_set2')
    generate_pattern_array_as_csv_with_indexes(length=1000, numSamples=5000, numClasses=2, percentError=3, filename='src/datasets/synthetic_set3')
    generate_pattern_array_as_csv_with_indexes(length=1000, numSamples=5000, numClasses=5, percentError=3, filename='src/datasets/synthetic_set4')

    f = open("data_cleaning_experiments_results.txt", 'a')
    f.write("Creating 4 datasets with 3% label error\n")
    f.write("Set one is 1000 sample of length 500 in 2 classes\n")
    f.write("Set one is 1000 sample of length 500 in 5 classes\n")
    f.write("Set one is 5000 sample of length 1000 in 2 classes\n")
    f.write("Set one is 5000 sample of length 1000 in 5 classes\n")

    f.flush()
    f.close()
