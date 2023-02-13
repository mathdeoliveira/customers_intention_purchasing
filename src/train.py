"""module responsible for training model"""
import os

import hydra
import mlflow
import pandas as pd
import structlog
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm

from split_data import SplitData
from utils import classification_metrics, load_train_data, name_category_features, save_model

logger = structlog.getLogger()

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('customer_intention')


def spliting_data() -> None:
    SplitData().split_data()


def to_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the columns in the dataframe to categorical data type.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the columns to be converted to categorical data type
    
    Returns:
    pd.DataFrame : DataFrame with the columns converted to categorical data type

    This function takes a DataFrame and converts the columns specified by the 
    name_category_features() function to categorical data type.
    """
    df[name_category_features()] = df[name_category_features()].astype(object)
    return df


def data_preparation() -> pd.DataFrame:
    """
    Prepare the data for modeling.
    
    Returns:
    Tuple: containing 4 dataframes (X_train, X_test, y_train, y_test) 
    that are splitted and prepared for modeling
    
    This function loads the train data, converts specified columns to categorical data type, 
    splits the data into training and testing sets, drops the 'to_split' column and returns the 
    data in 4 dataframes (X_train, X_test, y_train, y_test)
    """
    spliting_data()
    df = load_train_data()
    df[name_category_features()] = df[name_category_features()].astype(object)

    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=df[['to_split']],
    )
    X_train.drop('to_split', axis=1, inplace=True)
    X_test.drop('to_split', axis=1, inplace=True)

    return X_train, X_test, y_train, y_test


@hydra.main(config_path='../config/process', config_name='process')
def train(config: DictConfig):
    """
    Train different models and log the results in mlflow.

    Parameters:
    config : DictConfig : A configuration object that contains the enconding parameters and other settings
    
    This function prepares the data for modeling, applies transformation on the data and trains different models 
    (LogisticRegression, XGBClassifier, GradientBoostingRegressor, RandomForestClassifier and DecisionTreeClassifier)
    using transformed data and cross validation. It logs the results of each model in mlflow.
    """
    X_train, X_test, y_train, y_test = data_preparation()

    rare_enc = RareLabelEncoder(
        n_categories=config.enconding.rare_enc_n_categories, variables=list(config.enconding.rare_enc)
    )
    onehot_enc = OneHotEncoder(variables=list(config.enconding.onehot_enc))
    mm = MinMaxScaler()
    minmax_scaler = SklearnTransformerWrapper(
        transformer=mm, variables=list(config.enconding.minmax_scaler)
    )

    pipe = Pipeline(
        [('rare_enc', rare_enc), ('onehot', onehot_enc), ('minmax', minmax_scaler)]
    )
    
    X_train_transformed = pipe.fit_transform(X_train, y_train)
    X_test_transformed = pipe.transform(X_test)

    save_model(model=pipe, name_file='pipeline')

    logger.info("Starting training...")

    with mlflow.start_run(nested=True):
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'LGBMClassifier': lightgbm.LGBMClassifier()
        }

        for model_name, model in models.items():
            logger.info(f"Starting training in model: {model_name}")

            y_pred_train = cross_val_predict(model, X_train_transformed, y_train, cv=10)
            mlflow.log_param("model_name", model_name)

            metrics_train = {
                f"train_{metric}": value
                for metric, value in classification_metrics(
                    y_train, y_pred_train
                ).items()
            }

            mlmodel = model.fit(X_train_transformed, y_train)
            save_model(model=mlmodel, name_file=f'{model_name}_model')
            y_pred_test = mlmodel.predict(X_test_transformed)
            metrics_test = {
                f"test_{metric}": value
                for metric, value in classification_metrics(
                    y_test, y_pred_test
                ).items()
            }
            metrics = {**metrics_test, **metrics_train}
            
            mlflow.log_artifact(
                local_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f'../models/{model_name}_model.joblib',
                )
            )
            mlflow.log_artifact(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f'../models/pipeline.joblib',
                )
            )
            mlflow.log_metrics(metrics)
            mlflow.end_run()
            logger.info(f"Logged in mlflow for model: {model_name}")


if __name__ == "__main__":
    train()
