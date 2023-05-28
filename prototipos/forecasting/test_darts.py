import time
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.transformer_model import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import QuantileRegression

import reactivex as rx
from reactivex import operators as ops
from collections import OrderedDict
from csv import DictReader

import numpy as np

# CONSTANTES
token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()
write_api = client.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000))


#INSERTAR DATOS
data = rx \
    .from_iterable(DictReader(open('data.csv', 'r'))) \
    .pipe(ops.map(lambda row: parse_row(row)))

def parse_row(row: OrderedDict):
    return Point("views") \
        .tag("type", "value") \
        .field("y", float(row['y'])) \
        .time(row['ds'])

with InfluxDBClient(url="http://localhost:8086", token=token, org=org, debug=False) as client:
    with client.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000)) as write_api:
        write_api.write(bucket=bucket, record=data)

#OBTENER DATOS
query = 'from(bucket:"AGV")' \
        ' |> range(start:2007-12-10T15:00:00Z, stop:2016-01-20T15:00:00Z)'\
        ' |> filter(fn: (r) => r._measurement == "views")' \
        ' |> filter(fn: (r) => r._field == "y")' \
        ' |> filter(fn: (r) => r.type == "value")'
result = client.query_api().query(org=org, query=query)
raw = []
for table in result:
    for record in table.records:
        raw.append((record.get_value(), record.get_time()))
# print(raw[0:5])

print()
print("=== influxdb query into dataframe ===")
print()
df=pd.DataFrame(raw, columns=['y','ds'], index=None)
df['ds'] = df['ds'].values.astype('<M8[D]')
df.head()

series = TimeSeries.from_dataframe(df, "ds", "y", freq="D")
series = fill_missing_values(series=series)

#SEPARAR EN TRAIN IN VALIDATION

train, val = series[:-400], series[-400:]

scaler = Scaler()
train_scaled = scaler.fit_transform(train).astype(np.float32)
val_scaled = scaler.fit_transform(val).astype(np.float32)

# model = NBEATSModel(input_chunk_length=365*2, output_chunk_length=200, )
# model.fit(train_scaled, epochs=250, verbose=True)
# prediction = model.predict(series=train_scaled, n=365+100)

# model = FFT()
# model.fit(train_scaled)
# prediction = model.predict(n=365*2)

# model = RNNModel(
#     model="GRU",
#     input_chunk_length=200
# )
# model.fit(train_scaled, epochs=500)
# prediction = model.predict(n=365+50)

# model = AutoARIMA()
# model.fit(train_scaled)
# prediction = model.predict(n=30)

# model = ARIMA()
# model.fit(train_scaled)
# prediction = model.predict(n=365*2)

# model = Prophet()
# model.fit(train_scaled)
# prediction = model.predict(n=365*2)

# model = TCNModel(input_chunk_length=365*2,
#                  output_chunk_length=50,
#                 #  likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]))
#                 dropout=0.1)
# model.fit(train_scaled, epochs=400)
# # prediction = model.predict(n=365, num_samples=500)
# prediction = model.predict(n=365, mc_dropout=True, num_samples=500)

# model = ExponentialSmoothing()
# model.fit(train)
# pred = model.predict(n=365)

model = TransformerModel(input_chunk_length=365*3, output_chunk_length=31)
model.fit(train_scaled, epochs=100)
prediction = model.predict(series=train_scaled, n=400*2)

pred = scaler.inverse_transform(prediction)
            
# import matplotlib.pyplot as plt
            
# plt.figure(figsize=(10, 6))
# series.plot(label="actual (air)")
# pred.plot(label="forecast (air)")

pred_df = pred.pd_dataframe().reset_index()

# pred_df[['y_s0', 'ds']].tail()
# pred_df['measurement'] = "views"
# pred_df['mean'] = pred_df.mean(axis=1)

# print(pred_df)


# cp = pred_df[['ds', 'y_s0','measurement', 'mean']].copy()
# lines = [str(cp["measurement"][d]) 
#          + ",type=forecast" 
#          + " " 
#          + "y=" + str(cp["mean"][d])
#          + " " + str(int(time.mktime(cp['ds'][d].timetuple()))) + "000000000" for d in range(len(cp))]

pred_df[['y', 'ds']].tail()
pred_df['measurement'] = "views"

cp = pred_df[['ds', 'y','measurement']].copy()
lines = [str(cp["measurement"][d]) 
         + ",type=forecast" 
         + " " 
         + "y=" + str(cp["y"][d])
         + " " + str(int(time.mktime(cp['ds'][d].timetuple()))) + "000000000" for d in range(len(cp))]

_write_client = client.write_api(write_options=WriteOptions(batch_size=1000, 
                                                            flush_interval=10_000,
                                                            jitter_interval=2_000,
                                                            retry_interval=5_000))

_write_client.write(bucket, org, lines)

# plt.show()

_write_client.__del__()

client.__del__()