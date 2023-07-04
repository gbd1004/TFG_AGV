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
from darts.models.forecasting.nhits import NHiTSModel
from darts.metrics.metrics import mae, mase, dtw_metric

from util import test_univariate_ml, test_covariante, test_multivariate

torch.set_float32_matmul_precision('high')

VAL_SIZE = 300

plt.rcParams['figure.figsize'] = [15, 10]

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

    series_ed = get_series('encoder_derecho', df)[100:-154]
    series_ed = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ed))
    series_ei = get_series('encoder_izquierdo', df)[100:-154]
    series_ei = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ei))
    series_sr = get_series('out.set_speed_right', df)[100:-154]
    series_sl = get_series('out.set_speed_left', df)[100:-154]

    train_ed, val_ed = series_ed[:-VAL_SIZE], series_ed[-VAL_SIZE:]
    train_ei, val_ei = series_ei[:-VAL_SIZE], series_ei[-VAL_SIZE:]
    train_sr, val_sr = series_sr[:-VAL_SIZE], series_sr[-VAL_SIZE:]
    train_sl, val_sl = series_sl[:-VAL_SIZE], series_sl[-VAL_SIZE:]

    for salida in [50]:
        model = NHiTSModel(input_chunk_length=100, output_chunk_length=50)
        # model = NHiTSModel(
        #     input_chunk_length=53,
        #     output_chunk_length=50,
        #     num_stacks=4,
        #     num_blocks=4,
        #     num_layers=2,
        #     layer_widths=14,
        #     dropout=0.1855,
        #     activation="PReLU",
        #     MaxPool1d=False
        # )
        train_ed.plot()
        val_ed.plot()
        m_mae, m_mase, m_dtw, t_entrenamiento, t_prediccion = test_univariate_ml(model=model, nombre="nhits_univ", val_size=VAL_SIZE, salida=salida, series=series_ed, train_series=train_ed, val_series=val_ed)
        print(m_mae)
        print(m_mase)
        print(m_dtw)
        print(t_entrenamiento)
        print(t_prediccion)

        f.write("UNIVARIANTE " + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae) + "\n")
        f.write("MASE: " + str(m_mase) + "\n")
        f.write("DTW: " + str(m_dtw) + "\n")
        f.write("Tiempo entrenamiento " + str(t_entrenamiento) + "s\n")
        f.write("Tiempo prediccion " + str(t_prediccion) + "s\n\n")

        model = NHiTSModel(input_chunk_length=100, output_chunk_length=50)
        # model = NHiTSModel(
        #     input_chunk_length=53,
        #     output_chunk_length=50,
        #     num_stacks=4,
        #     num_blocks=4,
        #     num_layers=2,
        #     layer_widths=14,
        #     dropout=0.1855,
        #     activation="PReLU",
        #     MaxPool1d=False
        # )

        covariates_series = series_sr.stack(series_ei).stack(series_sl)
        train_covariates = covariates_series[:-VAL_SIZE]
        val_covariates = covariates_series[-VAL_SIZE:]
        train_ed.plot()
        val_ed.plot()
        m_mae, m_mase, m_dtw, t_entrenamiento, t_prediccion = test_covariante(model=model, nombre="nhits_cov", val_size=VAL_SIZE, salida=salida, series=series_ed, train_series=train_ed, val_series=val_ed, covariates_series=covariates_series, train_covariates=train_covariates, val_covariates=val_covariates)
        print(m_mae)
        print(m_mase)
        print(m_dtw)
        print(t_entrenamiento)
        print(t_prediccion)

        f.write("COVARIANTE " + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae) + "\n")
        f.write("MASE: " + str(m_mase) + "\n")
        f.write("DTW: " + str(m_dtw) + "\n")
        f.write("Tiempo entrenamiento " + str(t_entrenamiento) + "s\n")
        f.write("Tiempo prediccion " + str(t_prediccion) + "s\n\n")

        model = NHiTSModel(input_chunk_length=100, output_chunk_length=50)
        # model = NHiTSModel(
        #     input_chunk_length=53,
        #     output_chunk_length=50,
        #     num_stacks=4,
        #     num_blocks=4,
        #     num_layers=2,
        #     layer_widths=14,
        #     dropout=0.1855,
        #     activation="PReLU",
        #     MaxPool1d=False
        # )
        multivariate_series = series_ed.stack(series_ei).stack(series_sl).stack(series_sr)
        multivariate_train, multivariate_val = multivariate_series[:-VAL_SIZE], multivariate_series[-VAL_SIZE:]
        train_ed.plot()
        val_ed.plot()
        m_mae, m_mase, m_dtw, t_entrenamiento, t_prediccion = test_multivariate(model=model, nombre="nhits_mult", val_size=VAL_SIZE, salida=salida, series=multivariate_series, train_multi_series=multivariate_train, val_multi_series=multivariate_val, train_series=train_ed, val_series=val_ed)
        print(m_mae)
        print(m_mase)
        print(m_dtw)
        print(t_entrenamiento)
        print(t_prediccion)

        f.write("MULTIVARIANTE " + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae) + "\n")
        f.write("MASE: " + str(m_mase) + "\n")
        f.write("DTW: " + str(m_dtw) + "\n")
        f.write("Tiempo entrenamiento " + str(t_entrenamiento) + "s\n")
        f.write("Tiempo prediccion " + str(t_prediccion) + "s\n\n")

    f.close()
