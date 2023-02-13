"""module responsible for optimize training model"""
import os
from typing import Tuple

import lightgbm
import mlflow
import numpy as np
import optuna
import pandas as pd
import structlog
import yaml
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import f1_score

from train import data_preparation
from utils import load_model

logger = structlog.getLogger()
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), metric_name="f1"
)


def transformed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    This function applies data transformation to the training and testing datasets.
    The transformation is performed using a pipeline model that is loaded from a file.

    Returns:
    A tuple containing the transformed training and testing datasets, as well as the training and testing target variables.

    Example:
    X_train_transformed, X_test_transformed, y_train, y_test = transformed_data()
    """
    try:
        pipe = load_model('pipeline')
    except FileNotFoundError as e:
        logger.error(f"The pipeline model file does not exist: {e}")
        raise e

    try:
        X_train, X_test, y_train, y_test = data_preparation()

        X_train_transformed = pipe.transform(X_train)
        X_test_transformed = pipe.transform(X_test)

        logger.info("Data transformation completed successfully")

        return X_train_transformed, X_test_transformed, y_train, y_test
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise e


@mlflow_callback.track_in_mlflow()
def objective(trial):
    """
    Function to optimize the hyperparameters of a LightGBM classifier using Optuna optimization library.

    Parameters:
    trial (optuna.Trial): Optuna trial object that provides access to sampling methods and a storage area

    Returns:
    float: The F1 score of the LightGBM classifier with optimized hyperparameters on the test data.

    Notes:
    This function is expected to be used as the objective function in Optuna optimization. The Optuna library will
    repeatedly call this function to sample hyperparameters and evaluate the LightGBM classifier with these
    hyperparameters. The F1 score is used as the objective function to be minimized.
    """

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    X_train_transformed, X_test_transformed, y_train, y_test = transformed_data()

    train_data = lightgbm.Dataset(X_train_transformed, label=y_train)
    lgbm = lightgbm.train(params=param, train_set=train_data)
    y_pred_test = np.rint(
        lgbm.predict(X_test_transformed, num_iteration=lgbm.best_iteration)
    )
    f1 = round(f1_score(y_test, y_pred_test), 4)

    return f1


if __name__ == "__main__":
    study_name = "optimization_LGBM"
    direction = "maximize"
    n_trials = 2
    timeout = 600
    model_best_params_yaml_name = "model.yaml"

    study = optuna.create_study(study_name=study_name, direction=direction)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, callbacks=[mlflow_callback]
    )

    best_params = {"params": {**study.best_trial.params}}
    model_yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../config/model"
    )
    path_yaml = os.path.join(model_yaml_path, model_best_params_yaml_name)
        
    with open(path_yaml, "w") as file:
        yaml.safe_dump(best_params, file, default_flow_style=False)
