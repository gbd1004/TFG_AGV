from pandas import DataFrame
from influxdb_client import InfluxDBClient, Point, WriteOptions
import numpy as np

import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Diff
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.models.forecasting.transformer_model import TransformerModel
from darts.metrics.metrics import mae, mase, dtw_metric

torch.set_float32_matmul_precision('high')

VAL_SIZE = 320

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"

f = open("logs_transformer.txt", "w")

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

    scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()
    train_ed_scaled = scaler_ed.fit_transform(train_ed)
    val_ed_scaled = scaler_ed.transform(val_ed)
    series_ed_scaled = scaler_ed.transform(series_ed)
    train_ei_scaled = scaler_ei.fit_transform(train_ei)
    val_ei_scaled = scaler_ei.transform(val_ei)
    series_ei_scaled = scaler_ei.transform(series_ei)
    train_sr_scaled = scaler_sr.fit_transform(train_sr)
    val_sr_scaled = scaler_sr.transform(val_sr)
    series_sr_scaled = scaler_sr.transform(series_sr)
    train_sl_scaled = scaler_sl.fit_transform(train_sl)
    val_sl_scaled = scaler_sl.transform(val_sl)
    series_sl_scaled = scaler_sl.transform(series_sl)

    for salida in [1, 5, 50]:
        m_mae = 0
        m_mase = 0
        m_dtw = 0

        for i in range(0, 5):
            # model = TransformerModel(input_chunk_length=60, output_chunk_length=10)
            model = TransformerModel(
                input_chunk_length=60,
                output_chunk_length=10,
                d_model=128,
                nhead=4,
                num_encoder_layers=4,
                num_decoder_layers=3,
                dim_feedforward=128,
                dropout=0.02711,
                activation="SwiGLU",
                norm_type=None
            )

            pred_length = salida

            model.fit(train_ed_scaled, val_series=val_ed_scaled, epochs=200, verbose=False)
            prediction = model.predict(series=train_ed_scaled, n=pred_length)
            prediction = scaler_ed.inverse_transform(prediction)

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

            # model = TransformerModel(input_chunk_length=60, output_chunk_length=out_length)
            model = TransformerModel(
                input_chunk_length=60,
                output_chunk_length=out_length,
                d_model=128,
                nhead=4,
                num_encoder_layers=4,
                num_decoder_layers=3,
                dim_feedforward=128,
                dropout=0.02711,
                activation="SwiGLU",
                norm_type=None
            )

            pred_length = salida

            covariates = train_sr_scaled.stack(train_ei_scaled).stack(train_sl_scaled)
            val_covariates = series_sr_scaled.stack(series_ei_scaled).stack(series_sl_scaled)

            model.fit(series=train_ed_scaled, past_covariates=covariates, val_series=val_ed_scaled, val_past_covariates=val_covariates, epochs=200, verbose=False)
            prediction = model.predict(series=train_ed_scaled, past_covariates=covariates, n=pred_length)
            prediction = scaler_ed.inverse_transform(prediction)

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
            # model = TransformerModel(input_chunk_length=60, output_chunk_length=10)
            model = TransformerModel(
                input_chunk_length=60,
                output_chunk_length=10,
                d_model=128,
                nhead=4,
                num_encoder_layers=4,
                num_decoder_layers=3,
                dim_feedforward=128,
                dropout=0.02711,
                activation="SwiGLU",
                norm_type=None
            )

            multivariate_scaler = Scaler()
            multivariate_series = series_ed.stack(series_ei).stack(series_sl).stack(series_sr)
            multivariate_train, multivariate_val = multivariate_series[:-VAL_SIZE], multivariate_series[-VAL_SIZE:]
            multivariate_train_scaled = multivariate_scaler.fit_transform(multivariate_train).astype(np.float32)
            multivariate_val_scaled = multivariate_scaler.transform(multivariate_val).astype(np.float32)

            pred_length = salida

            model.fit(multivariate_train_scaled, val_series=multivariate_val_scaled, epochs=200, verbose=False)
            prediction = model.predict(series=multivariate_train_scaled, n=pred_length)
            prediction = multivariate_scaler.inverse_transform(prediction)
            prediction = prediction["encoder_derecho"]

            m_mae += mae(actual_series=val_ed[:pred_length], pred_series=prediction)
            m_mase += mase(actual_series=val_ed[:pred_length], pred_series=prediction, insample=train_ed)
            m_dtw += dtw_metric(actual_series=val_ed[:pred_length], pred_series=prediction)

        f.write("MULTIVARIANTE" + str(salida * 0.2) + "\n")
        f.write("MAE: " + str(m_mae / 5) + "\n")
        f.write("MASE: " + str(m_mase / 5) + "\n")
        f.write("DTW: " + str(m_dtw / 5) + "\n\n")

    f.close()