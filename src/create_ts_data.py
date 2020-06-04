import pandas as pd
import numpy as np
from utils.gen_ts_data import generate_pattern_data_as_csv

if __name__ == "__main__":
    generate_pattern_data_as_csv(length=500, numSamples=500, numClasses=3, percentError=3, filename='svm_test1')
    generate_pattern_data_as_csv(length=500, numSamples=500, numClasses=3, percentError=3, filename='svm_test2')
    generate_pattern_data_as_csv(length=500, numSamples=500, numClasses=3, percentError=3, filename='svm_test3')
