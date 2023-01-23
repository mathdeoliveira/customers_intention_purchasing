""" module responsible for loading data from raw directory"""
import os

import pandas as pd
import structlog

logger = structlog.getLogger()


class DataIngest:
    """Class Data Ingesting"""

    def __init__(self) -> None:
        """ """
        self.data_raw_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../data/raw"
        )
        self.data_raw_name = "data.csv"

    def load_data(self) -> pd.DataFrame:
        """Function for loading data from raw directory

        return:
        data from raw directory: pandas Dataframe
        """
        logger.info("Starting loading data...")

        loaded_data = pd.read_csv(self._path_raw_data())

        logger.info("Success loading data...")
        return loaded_data

    def _path_raw_data(self) -> str:
        """Function create raw data path

        return:
        raw data path: String
        """
        path_raw_data = os.path.join(self.data_raw_path, self.data_raw_name)

        return path_raw_data
