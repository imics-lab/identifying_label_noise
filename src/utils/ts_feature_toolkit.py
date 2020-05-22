#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This a small library of functions for extracting features from time series datagen

import scipy
from scipy.fft import fft
from scipy import signal
import numpy as np

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
    welch_f, welch_psd = signal.welch(X, fs=sample_rate);
    peak_psd = np.amax(welch_psd)
    energy = get_normalized_signal_energy(X)
    zcr = get_zero_crossing_rate(X)



    return [mean, stdev, energy, zcr]
