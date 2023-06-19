from pandas import DataFrame
from influxdb_client import InfluxDBClient, Point, WriteOptions
import time

import reactivex as rx
from reactivex import operators as ops
from collections import OrderedDict
from csv import DictReader

import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Diff
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.varima import VARIMA
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.random_forest import RandomForest
from darts.utils.statistics import plot_acf, plot_pacf
from darts.metrics import mape

import matplotlib.pyplot as plt
import torch

VAL_SIZE = 330

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()

def query_dataframe() -> DataFrame:
    query = 'from(bucket:"AGV")' \
        ' |> range(start:2023-05-30, stop:2023-05-31)' \
        ' |> aggregateWindow(every: 200ms, fn: last, createEmpty: false)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    df = query_api.query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="200ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

df = query_dataframe()

series_ed = get_series('encoder_derecho', df)


# plot_acf(series_ed, max_lag=100)
# plt.show()

series_ed = fill_missing_values(Diff(lags=10, dropna=False).fit_transform(series=series_ed))
series_ei = get_series('encoder_izquierdo', df)
series_ei = fill_missing_values(Diff(lags=10, dropna=False).fit_transform(series=series_ei))
series_sr = get_series('out.set_speed_right', df)
series_sl = get_series('out.set_speed_left', df)

train_ed, val_ed = series_ed[:-VAL_SIZE], series_ed[-VAL_SIZE:]


series = series_ed.stack(series_ei).stack(series_sr).stack(series_sl)

series_ed.plot()
series_sr.plot()
plt.show()


# train.plot()

scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()
scaler = Scaler()

series_ed_scaled = scaler_ed.fit_transform(series_ed).astype(np.float32)
series_ei_scaled = scaler_ei.fit_transform(series_ei).astype(np.float32)
series_sr_scaled = scaler_sr.fit_transform(series_sr).astype(np.float32)
series_sl_scaled = scaler_sl.fit_transform(series_sl).astype(np.float32)
series_scaled = scaler.fit_transform(series).astype(np.float32)

# train_scaled = scaler.fit_transform(train).astype(np.float32)
# val_scaled = scaler.fit_transform(val).astype(np.float32)
train_ed, val_ed = series_ed_scaled[:-VAL_SIZE], series_ed_scaled[-VAL_SIZE:]
train_ei, val_ei = series_ei_scaled[:-VAL_SIZE], series_ei_scaled[-VAL_SIZE:]
train_sr, val_sr = series_sr_scaled[:-VAL_SIZE], series_sr_scaled[-VAL_SIZE:]
train_sl, val_sl = series_sl_scaled[:-VAL_SIZE], series_sl_scaled[-VAL_SIZE:]
train_scaled, val_scaled = series_scaled[:-VAL_SIZE], series_scaled[-VAL_SIZE:]


# plot_acf(series_ed_scaled)
# plt.show()

# plot_pacf(series_ed_scaled)
# plt.show()

# print(series_ed_scaled)
# series_ed_scaled.plot()
# train_scaled.plot()
# val_scaled.plot()
# plt.legend()
# plt.show()

out = 200
# model = KalmanForecaster(dim_x=1000)
# model = Prophet()
# model = TransformerModel(
#     input_chunk_length=40,
#     output_chunk_length=10,
#     d_model=16,
#     nhead=2,
#     num_encoder_layers=4,
#     num_decoder_layers=1,
#     dim_feedforward=512,
#     dropout=0.48631234885236607,
#     activation="SwiGLU",
#     norm_type="LayerNorm"
# )
model = TransformerModel(input_chunk_length=40, output_chunk_length=10)
# model = TFTModel(input_chunk_length=360, output_chunk_length=20)
# model = NBEATSModel(input_chunk_length=100, output_chunk_length=out)
# model = NHiTSModel(input_chunk_length=40, output_chunk_length=10)
# model = RNNModel(input_chunk_length=100, model="LSTM")
# model = TCNModel(input_chunk_length=100, output_chunk_length=20)
# model = ARIMA()
# model = ARIMA(p=15, d=0, q=10)
# model = AutoARIMA()
# model = LinearRegressionModel(lags=10, lags_past_covariates=10, output_chunk_length=20)
# model = RandomForest(lags=100, lags_past_covariates=100, output_chunk_length=50, max_depth=50)
# model = VARIMA()
# model.fit(train_scaled, val_series=val_scaled, epochs=100)
# model.fit([train_ed, train_ei, train_sr, train_sl], epochs=20)
# model.fit([train_ed, train_ei], past_covariates=[train_sr, train_sl], epochs=100)

covariates = train_sr.stack(train_ei).stack(train_sl)
multivariate_scaler = Scaler()
multivariate_series = series_ed.stack(series_ei).stack(series_sl).stack(series_sr)
multivariate_series = multivariate_scaler.fit_transform(multivariate_series).astype(np.float32)
multivariate_train, multivariate_val = multivariate_series[:-VAL_SIZE], multivariate_series[-VAL_SIZE:]
# model.fit(train_ed, past_covariates=covariates)
# model.fit(multivariate_train)
model.fit(multivariate_train, val_series=multivariate_val, epochs=75)
# model.fit(train_ed)
# model.fit(train_scaled)
prediction = model.predict(series=multivariate_train, n=40)
# prediction = model.predict(series=train_scaled, n=VAL_SIZE)
# prediction = model.predict(series=train_scaled, n=out)
# prediction = model.predict(series=train_ed, past_covariates=covariates, n=50)
# prediction = model.predict(n=out)

# print(prediction)

# pred = scaler.inverse_transform(prediction)
pred = multivariate_scaler.inverse_transform(prediction)
# pred = prediction

# print(pred)

# series_ei.plot()
# series_sr.plot()
# series_ed.plot()
multivariate_scaler.inverse_transform(multivariate_series).plot()
# train.plot()
# val.plot()
pred.plot()
plt.show()

model.save("test_model.pt")

# historical = model.historical_forecasts(
#     multivariate_series, start=0.5, forecast_horizon=500, verbose=True
# )
# multivariate_series.plot()
# historical.plot()
# plt.show()
# print("MAPE = {:.2f}%".format(mape(historical, series)))