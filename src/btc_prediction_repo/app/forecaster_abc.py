# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod

import pandas as pd

from btc_prediction_repo.app.utils import resample_interpolate

logger = logging.getLogger(__name__)


class Forecaster(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.external_factors = []

    def select_ds_y_cols(self, ds: str, y: str) -> None:
        """Subselect ds and y columns for forecasting"""
        if ds not in self.df.columns or y not in self.df.columns:
            raise ValueError(f"Failed to find columns {ds=} or {y=} in df. Existing options: {self.df.columns}")
        self.df = self.df.rename(columns={ds: "ds", y: "y"})
        self.df = self.df[["ds", "y"]]
        self.df = self.df.drop_duplicates().reset_index(drop=True)

    def add_external_factor_rename_resample(self, external_df: pd.DataFrame, external_label: str) -> None:
        """External df must have ds and y columns"""
        if "ds" not in external_df.columns or "y" not in external_df.columns:
            raise ValueError(f"Please ensure the column names for external df are ds and y, not {external_df.columns}")
        external_df = external_df.rename(columns={"y": external_label})
        external_df = resample_interpolate(df=external_df, freq="D")
        self.df = self.df.merge(external_df, on="ds")
        self.external_factors.append(external_label)

    @abstractmethod
    def fit_model(self):
        pass
