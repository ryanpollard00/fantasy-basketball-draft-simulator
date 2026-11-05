"""
Train, evaluate and apply the fantasy point projection models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(actual, predicted):
    """Mean absolute error and root mean squared error, both in fantasy points."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }