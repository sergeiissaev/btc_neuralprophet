import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def multi_type_rand(x):
    if x == 1:
        return np.random.choice([False, True,])
    elif x == 2:
        return "auto"
    elif x == 3:
        return np.random.randint(0,100)

def create_np_args(ar: bool = False) -> dict:
    seasonality_mode = np.random.choice(["additive", "multiplicative"])
    growth = np.random.choice(["off", "linear", "discontinuous"])
    n_changepoints = np.random.choice([0, np.random.randint(1,100), np.random.randint(100, 10000)])
    changepoints_range = np.random.uniform(0, 1)
    trend_reg = np.random.choice([0, np.random.uniform(0, 1), np.random.randint(1, 100)])
    trend_reg_threshold = np.random.choice([False, True])
    yearly_seasonality = multi_type_rand(np.random.randint(1, 3))
    weekly_seasonality = multi_type_rand(np.random.randint(1, 3))
    seasonality_reg = np.random.choice([None, np.random.uniform(0, 1), np.random.randint(1, 100)])
    args = {"seasonality_mode": seasonality_mode, "growth": growth, "n_changepoints": n_changepoints,
     "changepoints_range": changepoints_range, "trend_reg": trend_reg, "trend_reg_threshold": trend_reg_threshold,
     "yearly_seasonality": yearly_seasonality, "weekly_seasonality": weekly_seasonality,
     "seasonality_reg": seasonality_reg}

    if ar:
        n_lags = np.random.choice([0, np.random.randint(0, 60)])
        ar_reg = np.random.choice([0, np.random.uniform(0, 1), np.random.randint(1, 100)])
        num_hidden_layers = np.random.choice([0, np.random.randint(1, 100)])
        args = args | {'n_lags': n_lags, 'n_forecasts': 30, 'ar_reg': ar_reg, 'num_hidden_layers': num_hidden_layers}

    return args

def create_df(asset_ticker: str, show_plot: bool = True) -> pd.DataFrame:
    ''' Obtain dataframe of dates to close price for an asset using yfinance  '''
    data = yf.download(asset_ticker)
    if show_plot:
        data['Adj Close'].plot()
        plt.savefig(f"data/article/{asset_ticker}_close.png")
        plt.show()
    df = data['Adj Close']
    df = df.reset_index()
    df = df.rename(columns={"Date": "ds", "Adj Close": "y"})
    return df

def select_ds_y_cols(df: pd.DataFrame, ds: str, y: str) -> pd.DataFrame:
    """Subselect ds and y columns for forecasting, and drops all duplicates"""
    if ds not in df.columns or y not in df.columns:
        raise ValueError(f"Failed to find columns {ds=} or {y=} in df")
    df = df.rename(columns={ds: "ds", y: "y"})
    df = df[["ds", "y"]]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def resample_interpolate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample data to daily and interpolate nans"""
    df = df.set_index("ds")
    df = df.resample(freq)
    df = df.interpolate(method="linear")
    df = df.reset_index()
    return df

def load_csv_file(file_path: Path, dtypes: List = None, date_column: str = None) -> pd.DataFrame:
    """Read in a CSV file with an optional dtypes list and a column name to convert to datetime"""
    if not file_path.is_file():
        err = FileNotFoundError(f"File at {file_path} does not exist. Are you running from root directory?")
        logger.error(err)
        raise err
    if date_column:
        date_column = [date_column]
    try:
        df = pd.read_csv(file_path, dtype=dtypes, parse_dates=date_column)
    except ValueError as e:
        df = pd.read_csv(file_path)
        err = ValueError(
            f"Passed column {date_column} is not a valid column name in the dataframe. Valid column names are {df.columns}"
        )
        raise err from e
    return df


def get_np_args(ar: bool):
    if ar:
        return ar_true.pop()
    return ar_false.pop()


ar_false=[create_np_args(ar=False) for _ in range(100)]
ar_true=[create_np_args(ar=True) for _ in range(100)]
