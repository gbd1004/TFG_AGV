import json
import os
import time

from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Diff
from darts.models.forecasting.transformer_model import TransformerModel
from darts.dataprocessing.transformers import Scaler
from influxdb_client import InfluxDBClient
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

def get_influxdb_credentials():
    token = os.getenv('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
    org = os.getenv('DOCKER_INFLUXDB_INIT_ORG')
    bucket = os.getenv('DOCKER_INFLUXDB_INIT_BUCKET')

    return token, org, bucket

def query_dataframe(client, bucket) -> DataFrame:
    query = 'from(bucket:"{bucket}")' \
        ' |> range(start: -26s, stop:now())' \
        ' |> aggregateWindow(every: 50ms, fn: last, createEmpty: true)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'.format(bucket=bucket)
    
    df = client.query_api().query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame, client, bucket) -> TimeSeries:
    for i in range(0, 25):
        try:
            series = TimeSeries.from_dataframe(df[-500:], "_time", tag, freq="50ms").astype(np.float32)
        except:
            time.sleep(0.5)
            df = query_dataframe(client=client, bucket=bucket)
            print(df)
            continue
        
    series = fill_missing_values(series=series)

    return series
    
def main():
    while True:
        # f = open("/forecasting/config.json")
        # config = json.load(f)

        # if config['load_model']:
        #     model_path = config['model_file']

        # model = TransformerModel.load(model_path)

        # token, org, bucket = get_influxdb_credentials()

        model = TransformerModel.load('../test_model.pt')

        token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
        bucket = "AGV"
        org = "TFG"

        with InfluxDBClient(url="http://localhost:8086", token=token, org=org, debug=False) as client:
            df = query_dataframe(client, bucket)

            series_ed = get_series('encoder_derecho', df, client, bucket)
            series_ed = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ed))
            series_ei = get_series('encoder_izquierdo', df, client, bucket)
            series_ei = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ei))
            series_sr = get_series('out.set_speed_right', df, client, bucket)
            series_sl = get_series('out.set_speed_left', df, client, bucket)

            scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()

            series_ed_scaled = scaler_ed.fit_transform(series_ed).astype(np.float32)
            series_ei_scaled = scaler_ei.fit_transform(series_ei).astype(np.float32)
            series_sr_scaled = scaler_sr.fit_transform(series_sr).astype(np.float32)
            series_sl_scaled = scaler_sl.fit_transform(series_sl).astype(np.float32)
            # series = series_ed.stack(series_ei).stack(series_sr).stack(series_sl)

            try:
                pred = model.predict(series=[series_sr_scaled, series_sl_scaled], past_covariates=[series_ed_scaled, series_ei_scaled], n=500)

                pred = scaler_sr.inverse_transform(pred[0])
                print(pred)
                series_sr.plot()
                pred.plot()
                plt.show()
            except:
                print("Error prediciendo")

if __name__ == "__main__":
    main()