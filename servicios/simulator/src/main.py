"""Servicio dedicado a simular el AGV. De momento solo son datos aleatorios"""

from datetime import datetime, timedelta
import socket
import random
import json
import time
import itertools
import csv

UDP_IP = "reciever"
UDP_PORT = 5004

# Funcion temporal para generar datos
def generar_datos():
    dato = {
        "time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
        "AGVID": 1,
        "???EncoderDerecho": random.randint(0, 1000000),
        "???EncoderIzquierdo": random.randint(0, 1000000),
        "In.CurrentL": random.randint(200, 240),
        "In.CurrentH": random.randint(200, 240),
        "In.I_MedidaBat": random.randint(0, 100),
        "In.guideError": random.random(),
        "Out.SetSpeedRight": random.randint(0, 32000),
        "Out.SetSpeedLeft": random.randint(0, 32000),
        "Out.Display": random.randint(0, 100)
    }

    return json.dumps(dato)

def simular_csv(sock, csv_path):
    time = datetime.utcnow()
    with open("/simulator/data/" + csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            delta = timedelta(seconds=float(row['time']))
            while datetime.utcnow() < time + delta:
                pass
            row['time'] = (time + delta).strftime('%Y-%m-%d %H:%M:%S.%f')
            row['AGVID'] = "Sim_1"
            sock.sendto(bytes(json.dumps(row), encoding="utf-8"), (UDP_IP, UDP_PORT))

def simular_aleatorio(sock):
    while True:
        datos = generar_datos()
        t_espera = random.randint(5, 20)
        time.sleep(t_espera / 1000)
        sock.sendto(bytes(datos, encoding="utf-8"), (UDP_IP, UDP_PORT))

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    f = open('/simulator/config.json')
    data = json.load(f)

    if data['from_csv']:
        if data['loop']:
            while True:
                simular_csv(sock, data['csv_file'])
        else:
            simular_csv(sock, data['csv_file'])
    else:
        simular_aleatorio(sock)