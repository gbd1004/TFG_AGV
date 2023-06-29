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

URL = "http://database:8086"

def get_influxdb_credentials():
    """Obtiene las credenciales de InfluxDB a partir de variables de entorno."""
    token = os.getenv('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
    org = os.getenv('DOCKER_INFLUXDB_INIT_ORG')
    bucket = os.getenv('DOCKER_INFLUXDB_INIT_BUCKET')

    return token, org, bucket


def query_dataframe(client, bucket, tiempo) -> DataFrame:
    """Obtiene un DataFrame con los datos de los últimos segundos.
    
    Argumentos:
    client -- cliente de la base de datos sobre la que hacer la consulta.
    bucket -- bucket en el que se encuentran los datos.
    tiempo -- número de segundos especificados para obtener los datos.
    """
    query = f'from(bucket:"{bucket}")' \
        f' |> range(start: -{tiempo}s)' \
        ' |> aggregateWindow(every: 200ms, fn: mean, createEmpty: true)' \
        ' |> filter(fn: (r) => r._measurement == "AGVDATA" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    dataframe = client.query_api().query_data_frame(query=query)
    dataframe.drop(columns=['result', 'table', '_start', '_stop'])
    dataframe.drop(dataframe.tail(1).index, inplace=True)

    dataframe['_time'] = dataframe['_time'].values.astype('<M8[ms]')
    dataframe.head()
    return dataframe


def get_series(tag: str, dataframe: DataFrame) -> TimeSeries:
    """Obtiene una TimeSeries concreta a partir de un DataFrame.
    
    tag -- nombre de la serie temporal a obtener.
    dataframe -- DataFrame del que se obtiene la serie temporal.
    """
    series = TimeSeries.from_dataframe(
        dataframe, "_time", tag, freq="200ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series


def get_all_series(dataframe):
    """Obtiene todasl las TimeSeries a utilizar a partir de un DataFrame.
    
    dataframe -- DataFrame del que se obtiene la serie temporal.
    """
    series_ed = get_series('encoder_derecho', dataframe)
    series_ed = fill_missing_values(
        Diff(lags=1, dropna=False).fit_transform(series=series_ed))
    series_ei = get_series('encoder_izquierdo', dataframe)
    series_ei = fill_missing_values(
        Diff(lags=1, dropna=False).fit_transform(series=series_ei))
    series_sr = get_series('out.set_speed_right', dataframe)
    series_sl = get_series('out.set_speed_left', dataframe)

    return series_ed, series_ei, series_sl, series_sr


def train_model(client, bucket, config):
    """Entrena el modelo de predicción.
    
    Argumentos:
    client -- cliente de la base de datos sobre la que hacer la consulta.
    bucket -- bucket en el que se encuentran los datos.
    config -- diccionario con la configuración del servicio.
    """
    tiempo = int(config['wait_time_before_train'])
    logging.info('Esperando %ss para datos para entrenamiento', tiempo)
    time.sleep(tiempo)

    print(bucket, tiempo)

    dataframe = query_dataframe(client, bucket, tiempo)

    series_ed, series_ei, series_sl, series_sr = get_all_series(dataframe)
    scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()

    series_ed_scaled = scaler_ed.fit_transform(series_ed).astype(np.float32)
    series_ei_scaled = scaler_ei.fit_transform(series_ei).astype(np.float32)
    series_sr_scaled = scaler_sr.fit_transform(series_sr).astype(np.float32)
    series_sl_scaled = scaler_sl.fit_transform(series_sl).astype(np.float32)

    covariates = series_sr_scaled.stack(series_sl_scaled)

    model = TransformerModel(input_chunk_length=100, output_chunk_length=50)

    logging.info("Iniciando entrenamiento")
    model.fit(series=[series_ed_scaled, series_ei_scaled],
              past_covariates=[covariates, covariates], epochs=200)

    model.save("/forecasting/model/" + config["model_file"] + ".pt")

    return scaler_ed, scaler_ei, scaler_sr, scaler_sl, model

def load_model(client, bucket, config):
    """Carga el modelo de predicción.
    
    Argumentos:
    client -- cliente de la base de datos sobre la que hacer la consulta.
    bucket -- bucket en el que se encuentran los datos.
    config -- diccionario con la configuración del servicio.
    """
    model_path = "/forecasting/model/" + \
    config['model_file'] + ".pt"
    model = TransformerModel.load(model_path)

    tiempo = int(config['wait_time_before_load'])
    logging.info(
        'Esperando %s segundos para datos para scaler', tiempo)
    time.sleep(tiempo)
    dataframe = query_dataframe(client, bucket, tiempo)
    series_ed, series_ei, series_sl, series_sr = get_all_series(
        dataframe)
    scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()
    scaler_ed.fit(series_ed)
    scaler_ei.fit(series_ei)
    scaler_sr.fit(series_sr)
    scaler_sl.fit(series_sl)

    return scaler_ed, scaler_ei, scaler_sr, scaler_sl, model

def main():
    """Funcion main."""
    logging.basicConfig(level=logging.INFO)
    with open("/forecasting/config.json", "r", encoding="utf8") as file:
        config = json.load(file)
    token, org, bucket_agv = get_influxdb_credentials()
    bucket_pred = "Predictions"

    time.sleep(0.5)

    with InfluxDBClient(url=URL, token=token, org=org, debug=False) as client:
        if config['load_model']:
            scaler_ed, scaler_ei, scaler_sr, scaler_sl, model = load_model(
                client, bucket_agv, config)
            logging.info("Modelo cargado satisfactoriamente")
        else:
            scaler_ed, scaler_ei, scaler_sr, scaler_sl, model = train_model(
                client, bucket_agv, config)
            logging.info("Modelo entrenado satisfactoriamente")

        while True:
            dataframe = query_dataframe(client, bucket_agv, 100)
            series_ed, series_ei, series_sl, series_sr = get_all_series(
                dataframe)

            series_ed_scaled = scaler_ed.transform(
                series_ed).astype(np.float32)
            series_ei_scaled = scaler_ei.transform(
                series_ei).astype(np.float32)
            series_sr_scaled = scaler_sr.transform(
                series_sr).astype(np.float32)
            series_sl_scaled = scaler_sl.transform(
                series_sl).astype(np.float32)

            covariates = series_sr_scaled.stack(series_sl_scaled)

            try:
                pred = model.predict(series=[series_ed_scaled, series_ei_scaled],
                                     past_covariates=[covariates, covariates], n=50)

                pred_ed = scaler_ed.inverse_transform(pred[0])
                pred_ei = scaler_ei.inverse_transform(pred[1])
                logging.info("Prediccion realizada correctamente")

                write_options = WriteOptions(
                    batch_size=500, flush_interval=100)
                with client.write_api(write_options=write_options) as write_api:
                    pred_ed = pred_ed.pd_dataframe().reset_index()
                    pred_ei = pred_ei.pd_dataframe().reset_index()

                    write_api.write(bucket=bucket_pred, record=pred_ed, write_precision='ms',
                                    data_frame_measurement_name='pred',
                                    data_frame_timestamp_column='_time')
                    write_api.write(bucket=bucket_pred, record=pred_ei, write_precision='ms',
                                    data_frame_measurement_name='pred',
                                    data_frame_timestamp_column='_time')
            except Exception as exc:
                logging.error("Error prediciendo")
                logging.error(exc)
            time.sleep(10)


if __name__ == "__main__":
    main()
