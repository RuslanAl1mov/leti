import numpy as np


def calculate_SNR(signal, noise):
    snr = 10*np.log10(np.var(signal)/np.var(noise))
    return snr


def return_noise_for_snr(signal, snr):
    desired_var = np.var(signal)/(10**(snr/10))
    return np.random.normal(scale=np.sqrt(desired_var), size=len(signal))
