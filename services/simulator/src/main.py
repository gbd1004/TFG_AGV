"""Servicio dedicado a simular el AGV"""

from datetime import datetime, timedelta
import logging
import socket
import random
import json
import sys
import time
import csv

UDP_IP = "reciever"
UDP_PORT = 5004

# Funcion temporal para generar datos


def generar_datos():
    """Genera datos adel AGV aleatorios."""
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
    """Simula el AGV leyendo datos de un archivo csv.
    
    Argumentos:
    sock -- socket UDP por el que se envían los datos.
    csv_path -- nombre del csv a utilizar.
    """
    actual_time = datetime.utcnow()
    with open("/simulator/data/" + csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            delta = timedelta(seconds=float(row['time']))
            while datetime.utcnow() < actual_time + delta:
                pass
            row['time'] = (actual_time + delta).strftime('%Y-%m-%d %H:%M:%S.%f')
            row['AGVID'] = "Sim_1"
            sock.sendto(bytes(json.dumps(row), encoding="utf-8"),
                        (UDP_IP, UDP_PORT))


def simular_aleatorio(sock):
    """Simula el AGV de manera aleatoria.
    
    Argumentos:
    sock -- socket UDP por el que se envían los datos.
    """
    while True:
        datos = generar_datos()
        t_espera = random.randint(5, 20)
        time.sleep(t_espera / 1000)
        sock.sendto(bytes(datos, encoding="utf-8"), (UDP_IP, UDP_PORT))


def main():
    """Función main."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    logging.basicConfig(level=logging.INFO)

    with open('/simulator/config.json', "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data['simulate']:
        logging.info("Simulación desactivada, saliendo...")
        sys.exit()

    if data['from_csv']:
        logging.info("Simulando desde CSV")
        if data['loop']:
            logging.info("Ejecución en bucle")
            while True:
                simular_csv(sock, data['csv_file'])
        else:
            logging.info("Ejecución única")
            simular_csv(sock, data['csv_file'])
    else:
        logging.info("Simulando aleatorio")
        simular_aleatorio(sock)


if __name__ == "__main__":
    main()
