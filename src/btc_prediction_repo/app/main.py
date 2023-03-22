# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf
# Import the plotting library
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet


# Get the data for the stock AAPL
data = yf.download('BTC-USD')


# Plot the close price of the AAPL
data['Adj Close'].plot()
plt.show()

df = data['Adj Close']

m = NeuralProphet()
m.data_freq = "D"
df = df.reset_index()
df = df.rename(columns={"Date": "ds", "Adj Close": "y"})



df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.2)

metrics_train = m.fit(df=df_train, freq="D")
metrics_test = m.test(df=df_test)

print(metrics_test)