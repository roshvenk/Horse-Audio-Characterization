# -*- coding: utf-8 -*-

# %% Import libraries

import scipy.fft as fft
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter


# %% Filtering

# this stuff copied from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# %% Basic processing, filtering and fourier transform

def signal_processing(filename, low_freq, high_freq):
    # stereo to mono conversion
    stereo_audio = AudioSegment.from_wav(filename)
    mono_audio = stereo_audio.set_channels(1)
    mono_audio.export("temp_audio.wav", format="wav")

    # get the signal, sample rate and time step size
    sample_rate, data = wavfile.read("temp_audio.wav")
    dt = 1/sample_rate

    # fourier transform of the whole signal
    x_fourier = fft.rfftfreq(len(data), d=dt)
    y_fourier = np.abs(fft.rfft(data))

    # filter the signal
    filtered = butter_bandpass_filter(data, low_freq, high_freq, sample_rate)
    filtered_fourier = np.abs(fft.rfft(filtered))

    return x_fourier, y_fourier, data, filtered, filtered_fourier, sample_rate


# %% Loop through each audio file, plotting the results and saving the filtered wav

for i in range(1, 9):
    name = "audio_" + str(i) + ".wav"

    # EDIT HERE FOR TESTING
    low_freq = 500
    high_freq = 10000
    # EDIT HERE FOR TESTING

    # process the original wav file
    x, y, signal, filt, filt_y, fs = signal_processing(
        name, low_freq, high_freq)

    # generate time array
    t_end = np.size(signal) / fs
    t = np.linspace(0, t_end, np.size(signal), endpoint=False)

    # plotting
    plt.plot(t, signal, 'r-')
    plt.xlabel("Time (s)")
    plt.title(name + " original signal")
    plt.show()

    plt.plot(t, filt, 'b-')
    plt.xlabel("Time (s)")
    plt.title(name + " filtered signal")
    plt.show()

    plt.plot(x, y, 'r-')
    plt.xlabel("Frequency (Hz)")
    plt.title(name + " original transform")
    plt.show()

    plt.plot(x, filt_y, 'b-')
    plt.xlabel("Frequency (Hz)")
    plt.title(name + " filtered transform")
    plt.show()

    # save filtered wav file
    name = "filtered_" + name
    wavfile.write(name, fs, filt.astype(np.int16))
