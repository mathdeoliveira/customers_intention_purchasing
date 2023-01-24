"""module responsible to split raw data into train and test"""
import os

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from data_ingest import DataIngest

logger = structlog.getLogger()
di = DataIngest()


class SplitData:
    def __init__(self) -> None:
        """
        Initialize the class.
        
        This function sets the following attributes:
        path : str : the directory where the csv files will be saved
        train_filename : str : name of the training data csv file
        test_filename : str : name of the testing data csv file
        full_data : pandas DataFrame : Dataframe of the original data
        """
        self.path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../data/processed"
        )
        self.train_filename = 'train.csv'
        self.test_filename = 'test.csv'
        self.full_data = di.load_data()

    def split_data(self) -> None:
        """
        Splits the dataset into training and testing sets, and saves them to csv files.
        
        This function takes the 'Month' and 'Revenue' columns from the original dataset and concatenates 
        them into a new column 'to_split'. Then it uses the 'to_split' column as the stratification criteria 
        to split the data into training and testing sets with a test size of 0.3 and random state 42. The 
        training set is saved to the file 'train_filename' and the testing set is saved to the file
        'test_filename'
        """
        self.full_data['to_split'] = self.full_data.Month.str.cat(
            self.full_data['Revenue'].astype(str), sep='_'
        )
        X_train, X_test = train_test_split(
            self.full_data,
            test_size=0.3,
            random_state=42,
            stratify=self.full_data[['to_split']],
        )
        logger.info("Starting splitting data...")
        self._to_csv(X_train, self.train_filename)
        self._to_csv(X_test, self.test_filename)

    def _to_csv(self, data: pd.DataFrame, filename: str) -> None:
        """
        Saves a DataFrame to a csv file.
        
        Parameters:
        data (pd.DataFrame): DataFrame to be saved
        filename (str): Name of the file to save the DataFrame to
        
        This function saves the data DataFrame to a csv file with the provided filename. The file is saved 
        in the directory specified by 'path' attribute of the class
        """
        data.to_csv(
            os.path.join(self.path, filename),
            index=False,
        )
