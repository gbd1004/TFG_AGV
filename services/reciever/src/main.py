"""Servicio Reciever. Inserta datos en la base de datos."""

import json
import os
import socket
import sys
import time
import logging
import influxdb_client

UDP_IP = "0.0.0.0"
UDP_PORT = 5004
URL = "http://database:8086"


def get_influxdb_credentials():
    token = os.getenv('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
    org = os.getenv('DOCKER_INFLUXDB_INIT_ORG')
    bucket = os.getenv('DOCKER_INFLUXDB_INIT_BUCKET')

    return token, org, bucket


def get_dbconn(token, org):
    client = influxdb_client.InfluxDBClient(
        url=URL,
        token=token,
        org=org
    )

    return client


def main():
    logging.basicConfig(level=logging.INFO)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    token, org, bucket = get_influxdb_credentials()

    with open('/reciever/config.json', "r", encoding="utf-8") as file:
        data = json.load(file)
    max_retries = int(data['max_retries'])

    connected = False
    retries = 0

    client = get_dbconn(token=token, org=org)

    with influxdb_client.InfluxDBClient(url=URL, token=token, org=org) as client:
        while not connected and retries < max_retries:
            try:
                write_options = influxdb_client.WriteOptions(
                    batch_size=1000, flush_interval=1_000, retry_interval=5_000)
                write_api = client.write_api(write_options=write_options)
                connected = True
            except Exception as exc:
                logging.warning(exc)
                retries += 1
                time.sleep(0.5)

        if not connected:
            logging.critical("Fallo con la conexión a la base de datos")
            sys.exit()

        logging.info(
            'Conexión a la base de datos realizada con éxito. Reintentos: %s', retries)

        while True:
            data, _ = sock.recvfrom(1024)
            data_json = json.loads(data)

            encoder_derecho = int(data_json['???EncoderDerecho'])
            if encoder_derecho & 0x80000000:
                encoder_derecho = -0x100000000 + encoder_derecho

            encoder_izquierdo = int(data_json['???EncoderIzquierdo'])
            if encoder_izquierdo & 0x80000000:
                encoder_izquierdo = -0x100000000 + encoder_izquierdo

            point = influxdb_client.Point("AGVDATA") \
                .tag("agvid", data_json['AGVID']) \
                .tag("type", "value") \
                .field("encoder_derecho", encoder_derecho) \
                .field("encoder_izquierdo", encoder_izquierdo) \
                .field("in.current_l", int(data_json['In.CurrentL'])) \
                .field("in.current_h", int(data_json['In.CurrentH'])) \
                .field("in.i_medida_bat", int(data_json['In.I_MedidaBat'])) \
                .field("in.guide_error", float(data_json['In.guideError'])) \
                .field("out.set_speed_right", int(data_json['Out.SetSpeedRight'])) \
                .field("out.set_speed_left", int(data_json['Out.SetSpeedLeft'])) \
                .field("out.display", int(data_json['Out.Display'])) \
                .time(data_json['time'])

            write_api.write(bucket=bucket, org=org, record=point)


if __name__ == "__main__":
    main()
