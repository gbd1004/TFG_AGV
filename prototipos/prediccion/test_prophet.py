from csv import DictReader
from prophet import Prophet
import pandas as pd
import time
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

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
print(raw[0:5])

print()
print("=== influxdb query into dataframe ===")
print()
df=pd.DataFrame(raw, columns=['y','ds'], index=None)
df['ds'] = df['ds'].values.astype('<M8[D]')
df.head()


m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast['measurement'] = "views"

cp = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper','measurement']].copy()
lines = [str(cp["measurement"][d]) 
         + ",type=forecast_prophet" 
         + " " 
         + "yhat=" + str(cp["yhat"][d]) + ","
         + "yhat_lower=" + str(cp["yhat_lower"][d]) + ","
         + "yhat_upper=" + str(cp["yhat_upper"][d])
         + " " + str(int(time.mktime(cp['ds'][d].timetuple()))) + "000000000" for d in range(len(cp))]

_write_client = client.write_api(write_options=WriteOptions(batch_size=1000, 
                                                            flush_interval=10_000,
                                                            jitter_interval=2_000,
                                                            retry_interval=5_000))

_write_client.write(bucket, org, lines)

_write_client.__del__()
client.__del__()