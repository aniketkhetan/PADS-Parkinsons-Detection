"""MiniRocket feature extraction.

Fitting is a separate step from transforming so the caller has to decide
explicitly what data the transform is allowed to see.
"""

import numpy as np

try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
except ImportError:  # sktime >= 0.30 moved these
    from sktime.transformations.panel.rocket._minirocket_multivariate import (
        MiniRocketMultivariate,
    )


def fit_minirocket(X, num_kernels=10000, seed=None, n_jobs=-1):
    """Fit on (n_samples, n_channels, n_timesteps). 10000 kernels -> 9996 features."""
    rocket = MiniRocketMultivariate(
        num_kernels=num_kernels, random_state=seed, n_jobs=n_jobs
    )
    rocket.fit(X)
    return rocket


def transform(rocket, X, batch_size=128):
    """Transform in batches to keep peak memory down."""
    out = []
    for start in range(0, X.shape[0], batch_size):
        chunk = rocket.transform(X[start:start + batch_size])
        out.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(out, axis=0)
