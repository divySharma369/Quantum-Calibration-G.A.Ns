import numpy as np

def apply_fft(pulse_signal, sampling_rate):
    """Analyze the frequency components of the generated pulses."""
    n = len(pulse_signal)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_values = np.fft.fft(pulse_signal)
    return freq, np.abs(fft_values)

def low_pass_filter(pulse_signal, cutoff, fs, order=5):
    """Simulate hardware bandwidth limitations."""
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, pulse_signal)
