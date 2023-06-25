import json
import socket
import sys
import time
import logging
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

UDP_IP = "0.0.0.0"
UDP_PORT = 5004
BUCKET = "AGV"
ORG = "TFG"
TOKEN = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
URL = "http://database:8086"

MAX_RETRIES = 10


def get_dbconn():
    client = influxdb_client.InfluxDBClient(
        url=URL,
        token=TOKEN,
        org=ORG
    )

    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    connected = False
    retries = 0

    client = get_dbconn()

    while not connected and retries < MAX_RETRIES:
        try:
            write_options = influxdb_client.WriteOptions(batch_size=1000, flush_interval=1_000, retry_interval=5_000)
            write_api = client.write_api(write_options=write_options)
            connected = True
        except:
            retries += 1
            time.sleep(0.5)

    if not connected:
        logging.critical("Fallo con la conexión a la base de datos")
        sys.exit()

    logging.info("Conexión a la base de datos realizada con éxito. Reintentos: " + str(retries))

    while True:
        data, addr = sock.recvfrom(1024)
        data_json = json.loads(data)
        
        ed = int(data_json['???EncoderDerecho'])
        if(ed & 0x80000000):
            ed = -0x100000000 + ed

        ei = int(data_json['???EncoderIzquierdo'])
        if(ei & 0x80000000):
            ei = -0x100000000 + ei

        point = influxdb_client.Point("AGVDATA") \
            .tag("agvid", data_json['AGVID']) \
            .tag("type", "value") \
            .field("encoder_derecho", ed) \
            .field("encoder_izquierdo", ei) \
            .field("in.current_l", int(data_json['In.CurrentL'])) \
            .field("in.current_h", int(data_json['In.CurrentH'])) \
            .field("in.i_medida_bat", int(data_json['In.I_MedidaBat'])) \
            .field("in.guide_error", float(data_json['In.guideError'])) \
            .field("out.set_speed_right", int(data_json['Out.SetSpeedRight'])) \
            .field("out.set_speed_left", int(data_json['Out.SetSpeedLeft'])) \
            .field("out.display", int(data_json['Out.Display'])) \
            .time(data_json['time'])
        
        write_api.write(bucket=BUCKET, org=ORG, record=point)