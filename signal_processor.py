# --- Imports ---
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

class SignalProcessor:
    """
    Handles all mathematical operations: Filtering and FFT calculations.
    """
    def __init__(self):
        pass

    def design_lowpass(self, cutoff, fs, order=5):
        """
        Generates Low-pass Butterworth filter coefficients (b, a).
        Returns (None, None) if parameters are invalid.
        """
        if fs is None or fs <= 0: 
            return None, None
        
        nyq = 0.5 * fs
        # Ensure cutoff is slightly less than Nyquist to avoid errors
        normal_cutoff = min(cutoff, nyq * 0.999) / nyq
        
        if normal_cutoff <= 0: 
            return None, None
            
        try:
            return signal.butter(order, normal_cutoff, btype='low', analog=False)
        except ValueError as e:
            print(f"Error creating lowpass filter: {e}")
            return None, None

    def design_highpass(self, cutoff, fs, order=5):
        """
        Generates High-pass Butterworth filter coefficients (b, a).
        Returns (None, None) if parameters are invalid.
        """
        if fs is None or fs <= 0: 
            return None, None
            
        nyq = 0.5 * fs
        normal_cutoff = min(cutoff, nyq * 0.999) / nyq
        
        if normal_cutoff <= 0: 
            return None, None
            
        try:
            return signal.butter(order, normal_cutoff, btype='high', analog=False)
        except ValueError as e:
            print(f"Error creating highpass filter: {e}")
            return None, None

    def compute_fft(self, data, fs, remove_dc=True):
        N = len(data)
        if N <= 1 or fs is None or fs <= 0:
            return np.array([]), np.array([])

        # 1. Remove DC (Keep this)
        if remove_dc:
            data_processed = data - np.mean(data)
        else:
            data_processed = data

        # 2. APPLY WINDOW FUNCTION (New Accuracy Fix)
        # Hanning window reduces spectral leakage
        window = np.hanning(N)
        data_processed = data_processed * window

        try:
            fft_output = fft(data_processed)
            T = 1.0 / fs
            freq_axis = fftfreq(N, T)
            
            # 3. CORRECTION FACTOR (Windowing reduces energy, we must compensate)
            # For Hanning window, the coherent gain loss is 0.5, so we multiply by 2.
            # Combined with the standard 2/N normalization:
            # Normalization = (2 / N) * (1 / 0.5) = 4 / N
            
            positive_freqs = freq_axis[:N // 2]
            
            # Use 4.0/N instead of 2.0/N to compensate for Hanning energy loss
            positive_magnitude = 4.0 / N * np.abs(fft_output[:N // 2])
            
            return positive_freqs, positive_magnitude
        except Exception as e:
            print(f"Error calculating FFT: {e}")
            return np.array([]), np.array([])