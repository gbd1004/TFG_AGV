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
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.varima import VARIMA
from darts.utils.statistics import plot_acf

import matplotlib.pyplot as plt
import torch

VAL_SIZE = 1000

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()

def query_dataframe() -> DataFrame:
    query = 'from(bucket:"AGV")' \
        ' |> range(start:2023-05-30, stop:2023-05-31)' \
        ' |> aggregateWindow(every: 50ms, fn: last, createEmpty: false)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    df = query_api.query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="50ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

df = query_dataframe()

series_ed = get_series('encoder_derecho', df)
series_ed = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ed))
series_ei = get_series('encoder_izquierdo', df)
series_ei = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ei))
series_sr = get_series('out.set_speed_right', df)
series_sl = get_series('out.set_speed_left', df)

series = series_ed.stack(series_ei).stack(series_sr).stack(series_sl)

train, val = series[:-VAL_SIZE], series[-VAL_SIZE:]

# train.plot()
val.plot()
plt.show()

scaler = Scaler()
train_scaled = scaler.fit_transform(train).astype(np.float32)
val_scaled = scaler.fit_transform(val).astype(np.float32)

# model = TransformerModel(input_chunk_length=120, output_chunk_length=60)
# model = NBEATSModel(input_chunk_length=240, output_chunk_length=120)
# model = RNNModel(input_chunk_length=120, model="LSTM", hidden_dim=100, n_rnn_layers=10)
model = TCNModel(input_chunk_length=120, output_chunk_length=60)
# model = ARIMA()
# model = VARIMA(p=5, d=1,q=10)
# model.fit(train_scaled, val_series=val_scaled, epochs=100)
# model.fit(train_scaled)
model.fit(series=train_scaled, val_series=val_scaled)
prediction = model.predict(series=train_scaled, n=240)
# model.fit(train_scaled)
# prediction = model.predict(series=train_scaled, n=VAL_SIZE)

pred = scaler.inverse_transform(prediction)
# pred = prediction

print(pred)

series.plot()
# train.plot()
# val.plot()
pred.plot()
plt.show()