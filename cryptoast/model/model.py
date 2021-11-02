"""
Objects for automatic descision making.
"""

from typing import Optional, Type, List, Tuple

import logging
import warnings
import tsfresh
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.backtesters import (
    BackTesterParent,
)  # , _get_percent_size, _return_fold_offsets

# pylint:disable=logging-format-interpolation, unused-argument

ALLOWED_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


def get_tsfresh_features(
    df: pd.DataFrame,
    time_col: str = "time",
    window: int = 30,
    progressbar: bool = True,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """Engineers tsfresh features over a given rolling window length.

    Args:
        df: Dataframe of the time-series with value and time columns.
        time_col: The time column. The dataframe needs to be sorted on this.
        window: Length of the window over which to compute features.
        progressbar: Whether to show the progressbar or not.
        n_jobs: Number of jobs to run in parrallel.
    """
    df = df.copy()
    value_col = [col for col in df if col != time_col][0]
    df = df[[time_col, value_col]]
    max_timeshift = window - 1
    df.insert(df.shape[1], "id", 0)
    common_params = dict(
        column_id="id",
        column_sort=time_col,
        n_jobs=n_jobs,
        disable_progressbar=not progressbar,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_rolled = tsfresh.utilities.dataframe_functions.roll_time_series(
            df, max_timeshift=max_timeshift, rolling_direction=1, **common_params
        )
    _, df_rolled.loc[:, "id"] = df_rolled.loc[:, "id"].str
    complete_window_length_mask = (
        df_rolled.id > sorted(df.loc[:, time_col].unique())[max_timeshift]
    )
    df_rolled = df_rolled.loc[complete_window_length_mask, :]
    df_extracted = tsfresh.extract_features(
        df_rolled, column_value=value_col, chunksize=None, **common_params
    )
    df_extracted = df_extracted.reset_index().rename(columns={"id": time_col})
    return df_extracted


class BackTesterDates(BackTesterParent):
    """Back tester for specific dates."""

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        test_dates: float,
        model_class: Type,
        steps_ahead: int,
        window: Optional[int],
        multi=True,
        **kwargs
    ):
        logging.info("Initializing train/test percentages")

        if len(test_dates) == 0:
            logging.error("Empty test dates sequence")
            raise ValueError("Invalid test dates")
        self.test_dates = test_dates
        if steps_ahead < 0:
            logging.error("Non positive steps ahead")
            raise ValueError("Invalid steps ahead")
        self.steps_ahead = steps_ahead
        if window and window <= 0:
            logging.error("Non positive window")
            raise ValueError("Invalid window")
        self.window = window
        self.expanding = False if self.window else True

        logging.info("Calling parent class constructor")
        super().__init__(error_methods, data, params, model_class, multi, **kwargs)

    def _get_index_from_date(self, date):
        return pd.Index(self.data.time).get_loc(date)

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""
        logging.info("Creating train test splits")
        train_splits = []
        test_splits = []
        safety_for_dot_errors = 1
        for test_date in self.test_dates:
            test_index = self._get_index_from_date(test_date)
            training_data_end = test_index - self.steps_ahead
            training_data_start = (
                0 if self.expanding else training_data_end - self.window
            )
            train_splits.append((training_data_start, training_data_end))
            test_splits.append((test_index, test_index + 1 + safety_for_dot_errors))
        return train_splits, test_splits
