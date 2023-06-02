from influxdb_client import InfluxDBClient, Point, WriteOptions
import time

import reactivex as rx
from reactivex import operators as ops
from collections import OrderedDict
from csv import DictReader

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.transformer_model import TransformerModel

VAL_SIZE = 5000

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()

# date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
data = rx \
    .from_iterable(DictReader(open('ETTh1.csv', 'r'))) \
    .pipe(ops.map(lambda row: parse_row(row)))

def parse_row(row: OrderedDict):
    return Point("etth") \
        .tag("type", "value") \
        .field("hufl", float(row['HUFL'])) \
        .field("hull", float(row['HULL'])) \
        .field("mufl", float(row['MUFL'])) \
        .field("mull", float(row['MULL'])) \
        .field("lufl", float(row['LUFL'])) \
        .field("lull", float(row['LULL'])) \
        .field("ot", float(row['OT'])) \
        .time(row['date'])

# with InfluxDBClient(url="http://localhost:8086", token=token, org=org, debug=False) as client:
with client.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000)) as write_api:
    write_api.write(bucket=bucket, record=data)

query = 'from(bucket:"AGV")' \
        ' |> range(start:2016-07-01, stop:2018-06-27)'\
        ' |> filter(fn: (r) => r._measurement == "etth")' \
        ' |> filter(fn: (r) => r._field == "ot")' \
        ' |> filter(fn: (r) => r.type == "value")'
result = query_api.query(org=org, query=query)
raw = []
for table in result:
    for record in table.records:
        raw.append((record.get_value(), record.get_time()))
print(raw[0:5])

print()
print("=== influxdb query into dataframe ===")
print()
df=pd.DataFrame(raw, columns=['ot','date'], index=None)
df['date'] = df['date'].values.astype('<M8[h]')
df.head()

series = TimeSeries.from_dataframe(df, "date", "ot", freq="H").astype(np.float32)
series = fill_missing_values(series=series)

train, val = series[:-VAL_SIZE], series[-VAL_SIZE:]

scaler = Scaler()
train_scaled = scaler.fit_transform(train).astype(np.float32)
val_scaled = scaler.fit_transform(val).astype(np.float32)

model = TransformerModel(input_chunk_length=24*14, output_chunk_length=12)
model.fit(train, val_series=val, epochs=100)
prediction = model.predict(series=train, n=24)

# model = Prophet()
# model.fit(train)
# prediction = model.predict(n=4000)

pred = prediction
# pred = scaler.inverse_transform(prediction)
pred_df = pred.pd_dataframe().reset_index()

pred_df[['date', 'ot']].tail()
pred_df['measurement'] = "etth"

cp = pred_df[['date', 'ot','measurement']].copy()
lines = [str(cp["measurement"][d]) 
         + ",type=forecast_single" 
         + " " 
         + "ot=" + str(cp["ot"][d])
         + " " + str(int(time.mktime(cp['date'][d].timetuple()))) + "000000000" for d in range(len(cp))]


with client.write_api(write_options=WriteOptions(batch_size=1000, flush_interval=10_000, jitter_interval=2000, retry_interval=5000)) as write_api:
    write_api.write(bucket, org, lines)

client.__del__()