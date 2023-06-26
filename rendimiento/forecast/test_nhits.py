from pandas import DataFrame
from influxdb_client import InfluxDBClient, Point, WriteOptions
import numpy as np

import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Diff
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.models.forecasting.nhits import NHiTSModel
from darts.metrics.metrics import mae, mase, dtw_metric

torch.set_float32_matmul_precision('high')

VAL_SIZE = 320

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"

f = open("logs_nhits.txt", "w")

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

    for salida in [1, 5, 50]:
        m_mae = 0
        m_mase = 0
        m_dtw = 0

        for i in range(0, 5):
            # model = NHiTSModel(input_chunk_length=60, output_chunk_length=10)
            model = NHiTSModel(
                input_chunk_length=53,
                output_chunk_length=10,
                num_stacks=4,
                num_blocks=4,
                num_layers=2,
                layer_widths=14,
                dropout=0.1855,
                activation="PReLu",
                MaxPool1d=False
            )

            pred_length = salida

            model.fit(train_ed, val_series=val_ed, epochs=200, verbose=False)
            prediction = model.predict(series=train_ed, n=pred_length)

            m_mae += mae(actual_series=val_ed[:pred_length], pred_series=prediction)
            m_mase += mase(actual_series=val_ed[:pred_length], pred_series=prediction, insample=train_ed)
            m_dtw += dtw_metric(actual_series=val_ed[:pred_length], pred_series=prediction)

        f.write("UNIVARIANTE" + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae / 5) + "\n")
        f.write("MASE: " + str(m_mase / 5) + "\n")
        f.write("DTW: " + str(m_dtw / 5) + "\n\n")
        
        m_mae = 0
        m_mase = 0
        m_dtw = 0

        for i in range(0, 5):
            if salida == 50:
                out_length = 50
            else:
                out_length = 10

            # model = NHiTSModel(input_chunk_length=60, output_chunk_length=out_length)
            model = NHiTSModel(
                input_chunk_length=53,
                output_chunk_length=out_length,
                num_stacks=4,
                num_blocks=4,
                num_layers=2,
                layer_widths=14,
                dropout=0.1855,
                activation="PReLu",
                MaxPool1d=False
            )

            pred_length = salida

            covariates = train_sr.stack(train_ei).stack(train_sl)
            val_covariates = series_sr.stack(series_ei).stack(series_sl)

            model.fit(series=train_ed, past_covariates=covariates, val_series=val_ed, val_past_covariates=val_covariates, epochs=200, verbose=False)
            prediction = model.predict(series=train_ed, past_covariates=covariates, n=pred_length)

            m_mae += mae(actual_series=val_ed[:pred_length], pred_series=prediction)
            m_mase += mase(actual_series=val_ed[:pred_length], pred_series=prediction, insample=train_ed)
            m_dtw += dtw_metric(actual_series=val_ed[:pred_length], pred_series=prediction)

        f.write("COVARIANTE " + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae / 5) + "\n")
        f.write("MASE: " + str(m_mase / 5) + "\n")
        f.write("DTW: " + str(m_dtw / 5) + "\n\n")

        m_mae = 0
        m_mase = 0
        m_dtw = 0

        for i in range(0, 5):
            # model = NHiTSModel(input_chunk_length=60, output_chunk_length=10)
            model = NHiTSModel(
                input_chunk_length=53,
                output_chunk_length=10,
                num_stacks=4,
                num_blocks=4,
                num_layers=2,
                layer_widths=14,
                dropout=0.1855,
                activation="PReLu",
                MaxPool1d=False
            )

            multivariate_series = series_ed.stack(series_ei).stack(series_sl).stack(series_sr)
            multivariate_train, multivariate_val = multivariate_series[:-VAL_SIZE], multivariate_series[-VAL_SIZE:]

            pred_length = salida

            model.fit(multivariate_train, val_series=multivariate_val, epochs=200, verbose=False)
            prediction = model.predict(series=multivariate_train, n=pred_length)
            prediction = prediction["encoder_derecho"]

            m_mae += mae(actual_series=val_ed[:pred_length], pred_series=prediction)
            m_mase += mase(actual_series=val_ed[:pred_length], pred_series=prediction, insample=train_ed)
            m_dtw += dtw_metric(actual_series=val_ed[:pred_length], pred_series=prediction)

        f.write("MULTIVARIANTE" + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae / 5) + "\n")
        f.write("MASE: " + str(m_mase / 5) + "\n")
        f.write("DTW: " + str(m_dtw / 5) + "\n\n")

    f.close()