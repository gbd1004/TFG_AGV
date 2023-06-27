import time
from pandas import DataFrame
from influxdb_client import InfluxDBClient, Point, WriteOptions
import numpy as np

import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Diff
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.models.forecasting.arima import ARIMA
from darts.metrics.metrics import mae, mase, dtw_metric

torch.set_float32_matmul_precision('high')

VAL_SIZE = 320

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"

f = open("logs_arima.txt", "w")

def query_dataframe(client) -> DataFrame:
    query = 'from(bucket:"AGV")' \
        ' |> range(start:2023-05-30, stop:2023-05-31)' \
        ' |> aggregateWindow(every: 200ms, fn: last, createEmpty: false)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    df = client.query_api().query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="200ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

with InfluxDBClient(url="http://localhost:8086", token=token, org=org) as client:
    df = query_dataframe(client)

    series_ed = get_series('encoder_derecho', df)
    series_ed = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ed))
    series_ei = get_series('encoder_izquierdo', df)
    series_ei = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ei))
    series_sr = get_series('out.set_speed_right', df)
    series_sl = get_series('out.set_speed_left', df)

    train_ed, val_ed = series_ed[:-VAL_SIZE], series_ed[-VAL_SIZE:]
    train_ei, val_ei = series_ei[:-VAL_SIZE], series_ei[-VAL_SIZE:]
    train_sr, val_sr = series_sr[:-VAL_SIZE], series_sr[-VAL_SIZE:]
    train_sl, val_sl = series_sl[:-VAL_SIZE], series_sl[-VAL_SIZE:]

    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    for salida in [1, 5, 50]:
        for i in range(0, 5):
            # model = ARIMA()
            model = ARIMA(p=16, d=1, q=0)
            start_fit = time.time()
            model.fit(train_ed)
            end_fit = time.time()
            t_entrenamiento += end_fit - start_fit

            pred_length = salida

            start_pred = time.time()
            prediction = model.predict(series=train_ed, n=pred_length)
            end_pred = time.time()
            t_prediccion = end_pred - start_pred

            m_mae += mae(actual_series=val_ed[:pred_length], pred_series=prediction)
            m_mase += mase(actual_series=val_ed[:pred_length], pred_series=prediction, insample=train_ed)
            m_dtw += dtw_metric(actual_series=val_ed[:pred_length], pred_series=prediction)

        f.write("UNIVARIABLE " + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae / 5) + "\n")
        f.write("MASE: " + str(m_mase / 5) + "\n")
        f.write("DTW: " + str(m_dtw / 5) + "\n")
        f.write("Tiempo entrenamiento " + str(t_entrenamiento / 5) + "s\n")
        f.write("Tiempo prediccion " + str(t_prediccion / 5) + "s\n\n")
    f.close()