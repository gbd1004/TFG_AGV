from collections import OrderedDict
from csv import DictReader
from influxdb_client import InfluxDBClient, Point, WriteOptions

import reactivex as rx
from reactivex import operators as ops

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"

data = rx \
    .from_iterable(DictReader(open('data.csv', 'r'))) \
    .pipe(ops.map(lambda row: parse_row(row)))

def parse_row(row: OrderedDict):
    return Point("views") \
        .tag("type", "value") \
        .field("y", float(row['y'])) \
        .time(row['ds'])

with InfluxDBClient(url="http://localhost:8086", token=token, org=org, debug=True) as client:
    with client.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000)) as write_api:
        write_api.write(bucket=bucket, record=data)