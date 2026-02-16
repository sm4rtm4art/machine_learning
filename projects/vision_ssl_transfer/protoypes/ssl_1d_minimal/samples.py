import math
import random

import numpy as np

"""
Generate randomized 1D samples with exaclty on distinct feature
"""

def generate(sample_length = 256, sample_number = 1024,n_features=4,noise_strength=0.05):
    positions = np.linspace(0, 1, sample_length)
    data = np.zeros([sample_number, sample_length])

    for i in range(sample_number):
        data[i] = generate_random_feature(positions,n_features,noise_strength)
    return data

"""
features are broad gaussian peak, sharp gaussian peak, oscillations, and no feature (pure noise) 
"""

def generate_random_feature(positions, n_features,noise_strength=0.05):
    values = np.zeros_like(positions)
    feature_id = random.randint(0, n_features)
    match feature_id:

        # Random Gaussian peak
        case 0:
            peak_position = 0.25 + 0.5 * np.random.random(1)
            peak_width = 0.05
            values += gaussian(positions, peak_position, peak_width)

        # Random sharp Gaussian peak
        case 1:
            peak_position = 0.25 + 0.5 * np.random.random(1)
            peak_width = 0.005
            values += gaussian(positions, peak_position, peak_width)

        # sinus
        case 2:
            phase = 2 * math.pi * np.random.random(1)
            frequency = 5
            values += sin(positions, frequency, phase)

    #    case > 2 is "no feature" -> only noise

    # Add noise
    values += noise_strength * np.random.randn(len(positions))

    return values.astype(np.float32)





def gaussian(positions, peak_position, peak_width):
    return np.exp(- (positions - peak_position) ** 2 / (2 * peak_width ** 2))

def sin(positions, frequency, phase):
    return np.sin(2 * math.pi * frequency * positions + phase)
