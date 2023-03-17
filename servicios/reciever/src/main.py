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
            write_api = client.write_api(write_options=SYNCHRONOUS)
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

        point = influxdb_client.Point("agv_logs") \
            .tag("vid", str(data_json["id"])) \
            .field("bateria", int(data_json["bateria"])) \
            .field("velocidad", int(data_json["velocidad"])) \
            .field("punto", data_json["punto"]) \
            .field("siguiente_punto", data_json["siguiente_punto"]) \
            .field("pos_x", int(data_json["pos_x"])) \
            .field("pos_y", int(data_json["pos_y"])) \
            .time(data_json["tiempo"])

        write_api.write(bucket=BUCKET, org=ORG, record=point)