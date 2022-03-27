import numpy as np
from scipy import signal
from utils.consts import *
import pandas as pd


def find_peaks(sig, fs):
    N, C = fs / 2, 1
    R = C / N
    b, a = signal.butter(8, R)
    filt = signal.filtfilt(b, a, sig)
    peaks, _ = signal.find_peaks(filt, height=0.25 * np.mean(np.abs(sig)), width=fs_RESP / 2)
    # plt.plot(sig)
    # plt.plot(therm_peaks, sig[therm_peaks], 'x')
    # plt.show()
    return peaks


def calc_median_peak_rates(y, fs):
    y_peaks = find_peaks(y, fs)
    y_deltas = y_peaks[1:] - y_peaks[:-1]
    ave_detla = 0.25 * np.mean(y_deltas) + 0.75 * np.median(y_deltas)
    ave_detla_seconds = ave_detla / fs
    try:
        rate = (1 / ave_detla_seconds) * 60
    except RuntimeWarning:
        rate = np.nan
    return rate


def butter_filter(sig, in_fs, cutoff_fs):
    N = 4
    Wn = cutoff_fs
    fs = in_fs

    sos = signal.butter(N, Wn, 'low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered


def filter_and_downsample(sig, in_fs, out_fs):
    if in_fs == out_fs:
        return sig
    if in_fs % out_fs > 0:
        raise ValueError("This function only works when in_fs is full divisible by out_fs")
    sig = butter_filter(sig, in_fs, out_fs / 2)
    subsamples = np.arange(0, len(sig), in_fs / out_fs).astype(np.int32)
    return sig[subsamples]


def calculate_weights(series: pd.Series):
    counts = series.value_counts().sort_index()
    weights = dict(min(counts) / counts)
    return weights
