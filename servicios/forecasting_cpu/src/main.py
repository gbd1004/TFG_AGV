import json
import logging
import os
import time

from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Diff
from darts.models.forecasting.transformer_model import TransformerModel
from darts.dataprocessing.transformers import Scaler
from influxdb_client import InfluxDBClient, WriteOptions
import numpy as np
from pandas import DataFrame

def get_influxdb_credentials():
    token = os.getenv('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
    org = os.getenv('DOCKER_INFLUXDB_INIT_ORG')
    bucket = os.getenv('DOCKER_INFLUXDB_INIT_BUCKET')

    return token, org, bucket

def query_dataframe(client, bucket, tiempo) -> DataFrame:
    query = 'from(bucket:"{bucket}")' \
        ' |> range(start: -{tiempo}s)' \
        ' |> aggregateWindow(every: 200ms, fn: mean, createEmpty: true)' \
        ' |> filter(fn: (r) => r._measurement == "AGVDATA" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'.format(bucket=bucket, tiempo=int(tiempo))
    
    df = client.query_api().query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])
    df.drop(df.tail(1).index, inplace=True)

    df['_time'] = df['_time'].values.astype('<M8[ms]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="200ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

def get_all_series(df):
    series_ed = get_series('encoder_derecho', df)
    series_ed = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ed))
    series_ei = get_series('encoder_izquierdo', df)
    series_ei = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ei))
    series_sr = get_series('out.set_speed_right', df)
    series_sl = get_series('out.set_speed_left', df)

    return series_ed, series_ei, series_sl, series_sr

def train_model(client, bucket, config):
    tiempo = float(config['wait_time_before_train'])
    logging.info("Esperando {t}s para datos para entrenamiento".format(t=tiempo))
    time.sleep(tiempo)

    # df = query_all_dataframe(client, bucket)
    df = query_dataframe(client, bucket, tiempo)

    series_ed, series_ei, series_sl, series_sr = get_all_series(df)
    scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()

    series_ed_scaled = scaler_ed.fit_transform(series_ed).astype(np.float32)
    series_ei_scaled = scaler_ei.fit_transform(series_ei).astype(np.float32)
    series_sr_scaled = scaler_sr.fit_transform(series_sr).astype(np.float32)
    series_sl_scaled = scaler_sl.fit_transform(series_sl).astype(np.float32)

    covariates = series_sr_scaled.stack(series_sl_scaled)

    model = TransformerModel(input_chunk_length=100, output_chunk_length=50)

    logging.info("Iniciando entrenamiento")
    model.fit(series=[series_ed_scaled, series_ei_scaled], past_covariates=[covariates, covariates], epochs=200)

    model.save("/forecasting_cpu/model/" + config["model_file"] + ".pt")

    return scaler_ed, scaler_ei, scaler_sr, scaler_sl, model

def main():
    logging.basicConfig(level=logging.INFO)
    f = open("/forecasting_cpu/config.json")
    config = json.load(f)
    token, org, bucket_agv = get_influxdb_credentials()
    bucket_pred = "Predictions"

    time.sleep(2)

    with InfluxDBClient(url="http://database:8086", token=token, org=org, debug=False) as client:
        if config['load_model']:
            model_path = "/forecasting_cpu/model/" + config['model_file'] + ".pt"
            model = TransformerModel.load(model_path)

            tiempo = float(config['wait_time_before_load'])
            logging.info("Esperando {t} segundos para datos para scaler".format(t=tiempo))
            time.sleep(tiempo)
            df = query_dataframe(client, bucket_agv, tiempo)
            series_ed, series_ei, series_sl, series_sr = get_all_series(df)
            scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()
            scaler_ed.fit(series_ed)
            scaler_ei.fit(series_ei)
            scaler_sr.fit(series_sr)
            scaler_sl.fit(series_sl)
            logging.info("Modelo cargado satisfactoriamente")
        else:
            scaler_ed, scaler_ei, scaler_sr, scaler_sl, model = train_model(client, bucket_agv, config)
            logging.info("Modelo entrenado satisfactoriamente")

        while True:
            df = query_dataframe(client, bucket_agv, 100)
            series_ed, series_ei, series_sl, series_sr = get_all_series(df)

            series_ed_scaled = scaler_ed.transform(series_ed).astype(np.float32)
            series_ei_scaled = scaler_ei.transform(series_ei).astype(np.float32)
            series_sr_scaled = scaler_sr.transform(series_sr).astype(np.float32)
            series_sl_scaled = scaler_sl.transform(series_sl).astype(np.float32)

            covariates = series_sr_scaled.stack(series_sl_scaled)

            try:
                pred = model.predict(series=[series_ed_scaled, series_ei_scaled], past_covariates=[covariates, covariates], n=50)

                pred_ed = scaler_ed.inverse_transform(pred[0])
                pred_ei = scaler_ei.inverse_transform(pred[1])
                logging.info("Prediccion realizada correctamente")
                with client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=100)) as write_api:
                    pred_ed = pred_ed.pd_dataframe().reset_index()
                    pred_ei = pred_ei.pd_dataframe().reset_index()

                    write_api.write(bucket=bucket_pred, record=pred_ed, write_precision='ms', data_frame_measurement_name='pred', data_frame_timestamp_column='_time')
                    write_api.write(bucket=bucket_pred, record=pred_ei, write_precision='ms', data_frame_measurement_name='pred', data_frame_timestamp_column='_time')
            except Exception as e:
                logging.error("Error prediciendo")
                logging.error(e)
            time.sleep(10)

if __name__ == "__main__":
    main()