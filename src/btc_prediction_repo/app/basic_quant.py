from btc_prediction_repo.app.main import NeuralProphetForecast
import plotly.io as pio
import pandas as pd

from btc_prediction_repo.app.utils import create_df

pio.renderers.default = "browser"


df = create_df(asset_ticker="AAPL", show_plot=True)
folds = create_split_dfs(df=df, split_percent=0.2, k=5)
best_args = {'seasonality_mode': 'multiplicative', 'growth': 'linear', 'n_changepoints': 52, 'changepoints_range': 0.224589916673666, 'trend_reg': 0.0, 'trend_reg_threshold': True, 'yearly_seasonality': False, 'weekly_seasonality': False, 'seasonality_reg': 17, 'n_lags': 25, 'n_forecasts': 30, 'ar_reg': 73.0, 'num_hidden_layers': 0}

model = fit_model(df=df[:-1], np_args=best_args)
forecast = predict_data(df=df, model=model, periods=500, n_historic_predictions=4800)
pred = forecast[forecast.ds==pd.Timestamp.now().normalize()]
true = df.iloc[-1].y
bought = forecast[forecast.ds == pd.Timestamp.now().normalize()-pd.Timedelta(days=1)].y.values[0]
profit = true - bought
print(f"{profit=}")
true_feed = true * 0.98
bought_feed = bought / 0.98
print(f"profit after 2% taker + 2% maker: {true_feed - bought_feed}")
plot_in_browser(model=model, data=forecast)