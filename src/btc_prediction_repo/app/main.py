import logging
from pathlib import Path
from statistics import mean
from typing import List



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

from btc_prediction_repo.app.dataclasses_file import Metrics
from btc_prediction_repo.app.forecaster_abc import Forecaster
from btc_prediction_repo.app.utils import multi_type_rand, get_np_args
from neuralprophet import NeuralProphet, set_random_seed
from tqdm import tqdm

set_random_seed(42)

logger = logging.getLogger(__name__)



class NeuralProphetForecast(Forecaster):
    def __init__(self, df: pd.DataFrame, ar: bool, np_args: dict = None):
        super().__init__(df=df)
        self.model = None
        self.np_args = np_args
        self.ar = ar

    def instantiate_np(self, retries: int = 0) -> None:
        if not self.np_args:
            err = ValueError("No arguments selected for np_args! Please call the .create_np_args() method first.")
            logger.error(err)
            raise err
        attempt_no = 0
        while attempt_no <= retries:
            try:
                assert not (self.np_args["n_lags"] == 0 and self.np_args["n_forecasts"] > 0)
                self.model = NeuralProphet(**self.np_args)
                logger.info(f"Instantiated model with args={self.np_args}")
                return
            except Exception as e:
                err = ValueError(f"Invalid set of arguments passed to np_args. Invalid configuration: {self.np_args}")
                logger.error(err)
                logger.warning("Randomly re-selecting parameters")
                logger.warning(e)
                self.create_np_args()
                attempt_no += 1
        raise ValueError(f"Gave up on trying to find fit after {retries=} times")

    def split_train_test(self, validation_fraction: float, frequency: str):
        return self.model.split_df(self.df, freq=frequency, valid_p=validation_fraction)

    def fit_model(self, df_train: pd.DataFrame = None, df_test: pd.DataFrame = None, plot: bool = False) -> None:
        ''' Fit model with df_train and df_test, or just the entire self.df, adding any lagged regressors if any '''
        set_random_seed(42)
        if df_train is None:
            df_train = self.df
            logger.info("Training on entire dataset.")
        self.model.add_lagged_regressor(names=self.external_factors)
        if plot and df_test is not None:
            _ = self.model.fit(df_train, freq="D", validation_df=df_test, progress="plot")
            plt.show()
        elif plot and df_test is None:
            raise ValueError("Please pass in a validation set in order to plot train and validation curves")
        elif df_test:
            print("Passed a df_test without selecting plot. df_test will be ignored.")
        try:
            _ = self.model.fit(df_train, freq="D")
        except Exception as e:
            err = ValueError("Incorrectly configured model. Cannot fit.")
            logger.error(err)
            raise err from e

    def graph_fit_on_train(self, df_train: pd.DataFrame):
        forecast = self.model.predict(df_train)
        forecast["yhat1"] = forecast["yhat1"].clip(lower=0)
        fig = self.model.plot(forecast)
        fig.show()

    def graph_fit_on_test(self, df_test: pd.DataFrame):
        forecast = self.model.predict(df_test)
        model = self.model.highlight_nth_step_ahead_of_each_forecast(1)
        fig = self.model.plot(forecast[-7 * 24 :])
        fig.show()

    def plot_parameters(self):
        fig_param = self.model.plot_parameters()
        fig_param.show()

    def predict_data(
        self, periods: int, n_historic_predictions: int, df: pd.DataFrame = None, plot_filename_stem: str = None
    ) -> pd.DataFrame:
        """Create a dataframe and optional plot containing predictions backwards n_historic_predictions and forwards periods rows"""
        if df is None:
            future = self.model.make_future_dataframe(
                self.df, periods=periods, n_historic_predictions=n_historic_predictions
            )
        else:
            future = self.model.make_future_dataframe(
                df, periods=periods, n_historic_predictions=n_historic_predictions
            )
        data = self.model.predict(future)
        data["yhat1"] = data["yhat1"].clip(lower=0)
        if plot_filename_stem is not None:
            fig = self.model.plot(data)
            fig.savefig(Path("data", "interim", f"{plot_filename_stem}.png"))
            fig.show()
        return data

    def get_mae(self, df: pd.DataFrame, yhat_steps: int) -> float:
        """Have the AI predict on a df and get mae compared to the ground truths"""
        yhat = f"yhat{yhat_steps}"
        try:
            preds = self.model.predict(df)
        except Exception as e:
            logger.error(f"Failed!  {yhat_steps=} {df=}")

            return -1
        preds = preds[preds[yhat].notna()]
        preds[yhat] = preds[yhat].clip(lower=0)
        return mean_absolute_error(preds["y"], preds[yhat])

    def plot_in_browser(self, data: pd.DataFrame) -> None:
        fig = self.model.plot(data)
        fig.show()


    def create_split_dfs(self, k: int) -> List[pd.DataFrame]:
        if self.model is None:
            raise ValueError("Must run instantiate_np() method prior to splitting dfs")
        return self.model.crossvalidation_split_df(self.df, k=k)

    def get_cv_mae(self, yhat_lookahead_for_metrics: int, folds: int, np_retries: int = 5, plot: bool = False) -> Metrics:
        ''' Run a cross validation for folds folds to get a metric (train and test) of how well these hyperparameters work '''
        self.instantiate_np(retries=np_retries)
        folds = self.create_split_dfs(k=folds)
        metrics_train = []
        metrics_val = []
        for df_train, df_val in tqdm(folds):
            self.fit_model(df_train=df_train)
            if self.ar:
                for yhat_steps in range(1, yhat_lookahead_for_metrics):
                    training_mae = self.get_mae(df=df_train, yhat_steps=yhat_steps)
                    val_mae = self.get_mae(df=df_val, yhat_steps=yhat_steps)
                    if training_mae == -1 or val_mae == -1:
                        # flag for unsolvable issue of sometimes .predict() fails with AR models
                        return Metrics(mae_train=-1, mae_val=-1)
                    metrics_train.append(training_mae)
                    metrics_val.append(val_mae)
            else:
                mae_train = self.get_mae(df=df_train, yhat_steps=1)
                mae_val = self.get_mae(df=df_val, yhat_steps=1)
                metrics_train.append(mae_train)
                metrics_val.append(mae_val)
            self.instantiate_np()
        self.fit_model(df_train=self.df)
        if plot:
            self.graph_fit_on_train(df_train=self.df)
        return Metrics(mae_train=mean(metrics_train), mae_val=mean(metrics_val))

    def create_np_args(self) -> None:
        self.np_args = get_np_args(ar=self.ar)





    def plot_data(self, title: str = "Data"):
        plt.plot(self.df["ds"], self.df["y"])
        plt.title(title)
        plt.xlabel("Time")
        plt.show()
