# -*- coding: utf-8 -*-
import logging

from btc_prediction_repo.app.main import NeuralProphetForecast
from btc_prediction_repo.app.utils import create_df
from btc_prediction_repo.config.config_model import NP_ARGS

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
    neural = NeuralProphetForecast(df=create_df(asset_ticker="AAPL", show_plot=False), ar=True, np_args=NP_ARGS)
    neural.instantiate_np(retries=0)
    neural.fit_model()
    data = neural.predict_data(periods=100, n_historic_predictions=100, plot_filename_stem="AAPL_preds")
    print(data[data.yhat1.notna()].iloc[-1])
    print(data[data.yhat2.notna()].iloc[-1])
    print(data[data.yhat3.notna()].iloc[-1])
