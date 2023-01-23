""""""
import os

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from data_ingest import DataIngest

logger = structlog.getLogger()
di = DataIngest()


class SplitData:
    def __init__(self) -> None:
        self.path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../data/processed"
        )
        self.train_filename = 'train.csv'
        self.test_filename = 'test.csv'
        self.full_data = di.load_data()

    def split_data(self) -> None:
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
        data.to_csv(
            os.path.join(self.path, filename),
            index=False,
        )
