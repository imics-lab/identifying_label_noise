#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This a small library of functions for extracting features from time series datagen

import scipy
from scipy.fft import fft
from scipy import signal
import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc
#from tsfresh.feature_extraction.feature_calculators import abs_energy, absolute_sum_of_changes, benford_correlation, count_above_mean, count_below_mean, kurtosis, longest_strike_above_mean, number_crossing_m, number_peaks, sample_entropy
from tsfresh.utilities.dataframe_functions import impute

def get_normalized_signal_energy(X):
    return np.mean(np.square(X))

def get_zero_crossing_rate(X):
    mean = np.mean(X)
    zero_mean_signal = np.subtract(X, mean)
    return np.mean(np.absolute(np.edif1d(np.sign(X))))

def get_features_from_one_signal(X, sample_rate=50):
    assert X.ndim ==1, "Expected single signal in feature extraction"
    mean = np.mean(X)
    stdev = np.std(X)
    abs_energy = fc.abs_energy(X)
    sum_of_changes = fc.absolute_sum_of_changes(X)
    benford = fc.autocorrelation(X, sample_rate)
    count_above_mean = fc.count_above_mean(X)
    count_below_mean = fc.count_below_mean(X)
    kurtosis = fc.kurtosis(X)
    longest_above = fc.longest_strike_above_mean(X)
    zero_crossing = fc.number_crossing_m(X, mean)
    num_peaks = fc.number_peaks(X, int(sample_rate/10))
    sample_entropy = fc.sample_entropy(X)

    return [mean,
        stdev,
        abs_energy,
        sum_of_changes,
        benford,
        count_above_mean,
        count_below_mean,
        kurtosis,
        longest_above,
        zero_crossing,
        num_peaks,
        sample_entropy]

def get_features_for_set(X, sample_rate=50, num_samples=100):
    sample_length = len(X[0])
    fet = np.zeros((num_samples, 12))
    for i in range(num_samples):
        fet[i, :] = get_features_from_one_signal(X[i, :])
    return fet
