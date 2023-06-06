import csv
from datetime import datetime, timedelta

from collections import OrderedDict
from influxdb_client import InfluxDBClient, Point, WriteOptions

import reactivex as rx
from reactivex import operators as ops

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"

time = datetime(2023,5,30,0,0,0,0)

def parse_row(row: OrderedDict):
    global time

    delta = timedelta(seconds=float(row['time']))
    _time = time + delta

    ed = int(row['???EncoderDerecho'])
    if(ed & 0x80000000):
        ed = -0x100000000 + ed

    return Point("test") \
        .tag("type", "value") \
        .field("encoder_derecho", ed) \
        .field("encoder_izquierdo", int(row['???EncoderIzquierdo'])) \
        .field("in.current_l", int(row['In.CurrentL'])) \
        .field("in.current_h", int(row['In.CurrentH'])) \
        .field("in.i_medida_bat", int(row['In.I_MedidaBat'])) \
        .field("in.guide_error", float(row['In.guideError'])) \
        .field("out.set_speed_right", int(row['Out.SetSpeedRight'])) \
        .field("out.set_speed_left", int(row['Out.SetSpeedLeft'])) \
        .field("out.display", int(row['Out.Display'])) \
        .time(_time)

def main():
    data = rx \
        .from_iterable(csv.DictReader(open('datos_agv.csv', 'r'))) \
        .pipe(ops.map(lambda row: parse_row(row)))
    
    
    with InfluxDBClient(url="http://localhost:8086", token=token, org=org, debug=True) as client:
        with client.write_api(write_options=WriteOptions(batch_size=50_000, flush_interval=10_000)) as write_api:
            write_api.write(bucket=bucket, record=data, write_precision='us')

if __name__ == "__main__":
    main()