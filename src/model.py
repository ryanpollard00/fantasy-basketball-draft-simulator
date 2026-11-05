"""
Train, evaluate and apply the fantasy point projection models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

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


def train_xgboost(train):
    """Fit an XGBoost model on the training rows using config.XGB_PARAMS."""
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def season_splits(train):
    """Expanding-window splits: fit on earlier training seasons, validate on the next one."""
    seasons = config.TRAIN_TARGET_SEASONS
    splits = []
    for i in range(1, len(seasons)):
        fit_rows = np.flatnonzero(train["SEASON"].isin(seasons[:i]))
        validation_rows = np.flatnonzero(train["SEASON"] == seasons[i])
        splits.append((fit_rows, validation_rows))
    return splits


def tune(model, param_grid, train):
    """Try every combination in param_grid with season-based cross-validation.

    Returns the fitted search; its best model is refit on all training rows.
    """
    search = GridSearchCV(
        model,
        param_grid,
        cv=season_splits(train),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(train[MODEL_FEATURES], train[TARGET])
    return search


def tune_random_forest(train):
    return tune(RandomForestRegressor(**config.RF_PARAMS), config.RF_PARAM_GRID, train)


def tune_xgboost(train):
    return tune(XGBRegressor(**config.XGB_PARAMS), config.XGB_PARAM_GRID, train)