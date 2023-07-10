# -*- coding: utf-8 -*-
import logging

from btc_prediction_repo.app.main import NeuralProphetForecast
from btc_prediction_repo.app.utils import create_df
from btc_prediction_repo.config.config_model import NP_ARGS

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
    neural = NeuralProphetForecast(df=create_df(asset_ticker="AAPL", show_plot=False), ar=True, np_args=NP_ARGS)
    logger.info(f"Selected np args {neural.np_args}")
    neural.instantiate_np(retries=0)
    neural.fit_model()
    neural.predict_data(periods=100, n_historic_predictions=100, plot_filename_stem="AAPL_preds")
