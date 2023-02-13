"""Module responsible for training the final model.

This module uses the `transformed_data` module to retrieve the preprocessed training data,
trains a LightGBM model using the provided configuration parameters, and saves the trained
model using the `save_model` utility.
"""
import hydra
import lightgbm
import structlog
from omegaconf import DictConfig

from optimization import transformed_data
from utils import save_model

logger = structlog.getLogger()


@hydra.main(config_path='../config/model', config_name='model')
def final_model(config: DictConfig):
    """Trains the final model and saves it.

    Args:
        config (DictConfig): Configuration parameters for the model.

    Returns:
        None
    """
    X_train_transformed, _, y_train, _ = transformed_data()

    lgbm = lightgbm.LGBMClassifier(
        bagging_fraction=config.params.bagging_fraction,
        bagging_freq=config.params.bagging_freq,
        feature_fraction=config.params.feature_fraction,
        lambda_l1=config.params.lambda_l1,
        lambda_l2=config.params.lambda_l2,
        min_child_samples=config.params.min_child_samples,
        num_leaves=config.params.num_leaves,
    )
    lgbm.fit(X_train_transformed, y_train)
    save_model(lgbm, 'final_model')


if __name__ == "__main__":
    final_model()
