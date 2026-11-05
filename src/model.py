"""
Train, evaluate and apply the fantasy point projection models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import config
from src.features import MODEL_FEATURES, TARGET


def evaluate(actual, predicted):
    """Mean absolute error and root mean squared error, both in fantasy points."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def train_random_forest(train):
    """Fit a Random Forest on the training rows using config.RF_PARAMS."""
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model