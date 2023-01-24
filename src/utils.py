"""module responsible for utils functions"""
import os

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def name_category_features() -> list:
    """
    Returns the names of the categorical features.
    
    Returns:
    list : list of strings containing the names of the categorical features
    
    This function returns the names of the categorical features of the dataset.
    """
    return [
        'OperatingSystems',
        'Browser',
        'Region',
        'TrafficType',
        'VisitorType',
        'Weekend',
    ]


def load_train_data() -> pd.DataFrame:
    """
    Loads the training data.
    
    Returns:
    pd.DataFrame : DataFrame containing the training data
    
    This function loads the training data from a csv file 
    located in the directory specified by 'path' and 'file_name' attributes.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/processed")
    file_name = 'train.csv'

    df = pd.read_csv(os.path.join(path, file_name))
    return df


def classification_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    """
    Compute classification metrics.
    
    Parameters:
    actual (pd.Series) : Series containing the actual values
    pred (pd.Series) : Series containing the predicted values
    
    Returns:
    dict : A dictionary containing recall, precision, F1 and accuracy scores
    
    This function computes classification metrics (Recall, Precision, F1 and Accuracy) 
    for the given actual and predicted values.
    """
    return {
        "Recall": recall_score(actual, pred),
        "Precision": precision_score(actual, pred),
        "F1": f1_score(actual, pred),
        "Accuracy": accuracy_score(actual, pred),
    }

def save_model(model, name_file: str):
    """Save model in .joblib
    Args:
        model: model to save in joblib
        name_file: str, name for file to be save
    """
    mdir = _model_dir()
    joblib.dump(model, f'{mdir}/{name_file}.joblib')
    
def _model_dir() -> str:
    """Return the directory models
    Returns:
        model_dir: str, models directory
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
    return model_dir