import numpy as np
from .letter_sequence_to_numeric import letter_sequence_to_numeric

def get_power_array(k):
    return np.power(4, np.arange(0, k))

def read_kmers(read, power_array=None):
    numeric = letter_sequence_to_numeric(read)
    return np.convolve(numeric, power_array, mode='valid')  # % 452930477

