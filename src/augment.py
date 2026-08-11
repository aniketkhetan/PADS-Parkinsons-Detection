"""Time-series augmentation: jittering and magnitude scaling."""

import numpy as np


def add_jitter(ts, sigma=0.01):
    """Add Gaussian noise to time series."""
    return ts + np.random.normal(loc=0, scale=sigma, size=ts.shape)


def magnitude_scaling(ts, alpha=0.1):
    """Scale time series by a random factor."""
    factor = np.random.uniform(1 - alpha, 1 + alpha)
    return ts * factor


def augment_dataset(X, y, sigma=0.01, alpha=0.1, seed=None):
    """Double the dataset by appending one synthetic copy of every sample.

    X is (n_samples, n_channels, n_timesteps). Returns the doubled arrays plus
    a source index mapping each row back to the recording it derives from;
    rows sharing a source index are the same subject and must not straddle a
    train/test split.
    """
    if seed is not None:
        np.random.seed(seed)

    # Jitter and scaling are applied to alternating halves of the data.
    synthetic = np.empty_like(X)
    for i in range(X.shape[0]):
        if i % 2 == 0:
            synthetic[i] = add_jitter(X[i], sigma=sigma)
        else:
            synthetic[i] = magnitude_scaling(X[i], alpha=alpha)

    X_aug = np.concatenate([X, synthetic], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    source_index = np.concatenate([np.arange(X.shape[0]), np.arange(X.shape[0])])
    return X_aug, y_aug, source_index
