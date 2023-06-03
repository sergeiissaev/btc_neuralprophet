import logging
from pathlib import Path

from btc_prediction_repo.app.main import NeuralProphetForecast
from btc_prediction_repo.app.utils import load_csv_file, create_df
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
    neural = NeuralProphetForecast(df=create_df(asset_ticker="AAPL", show_plot=False), ar=True)
    metrics_list = []
    params_list = []
    for _ in range(2):
        neural.create_np_args()
        logger.info(f"Selected np args {neural.np_args}")
        metrics = neural.get_cv_mae(yhat_lookahead_for_metrics=14, folds=10, np_retries=32)
        metrics_list.append(metrics)
        params_list.append(neural.np_args)

    for metric, param in zip(metrics_list, params_list):
        print()
        print(f"{metric=}\n{param=}")