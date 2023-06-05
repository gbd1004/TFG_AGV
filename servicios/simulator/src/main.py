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


class Punto:
    nueva_id = itertools.count()

    def __init__(self, x, y):
        self.id = next(Punto.nueva_id)
        self.x = x
        self.y = y

    def __str__(self):
        return "[{0}: x:{1}, y:{2}]".format(self.id, self.x, self.y)


class Mapa:
    def __init__(self, lista_puntos: list()):
        self.lista_puntos = lista_puntos


class AGV:
    nueva_id = itertools.count()

    def __init__(self, punto_inicial: Punto, siguiente: Punto):
        self.id = next(AGV.nueva_id)
        self.bateria = 100.0
        self.velocidad = 0.05
        self.punto = punto_inicial
        self.siguiente_punto = siguiente
        self.pos_x = punto_inicial.x
        self.pos_y = punto_inicial.y

    def log(self, sock):
        datos = {
            "tiempo": str(datetime.utcnow()),
            "id": self.id,
            "bateria": self.bateria,
            "velocidad": self.velocidad,
            "punto": str(self.punto),
            "siguiente_punto": str(self.siguiente_punto),
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
        }
        print(datos)
        sock.sendto(bytes(json.dumps(datos), encoding="utf-8"), (UDP_IP, UDP_PORT))

    def actualizar_bateria(self, dt):
        if self.bateria > 0:
            self.bateria -= dt / 10
        else:
            self.bateria = 0

    def actualizar_posicion(self, dt):
        if self.siguiente_punto is not None and self.bateria > 0:
            dir_x = self.siguiente_punto.x - self.punto.x
            dir_y = self.siguiente_punto.y - self.punto.y

            self.pos_x = self.pos_x + dir_x * self.velocidad * dt
            self.pos_y = self.pos_y + dir_y * self.velocidad * dt

            if abs(self.pos_x - self.siguiente_punto.x) < 0.001 and \
               abs(self.pos_y - self.siguiente_punto.y) < 0.001:
                self.punto = self.siguiente_punto
                self.pos_x = self.siguiente_punto.x
                self.pos_y = self.siguiente_punto.y
                self.siguiente_punto = None

    def simular(self, sock):
        dt = 0
        tiempo_log = 5

        while True:
            start = time.time()
            tiempo_log -= dt

            self.actualizar_bateria(dt)
            self.actualizar_posicion(dt)

            if tiempo_log < 0:
                self.log(sock)
                tiempo_log = 5

            end = time.time()
            dt = end - start


# Funcion temporal para generar datos
def generar_datos():
    """Genera datos aleatorios del AGV con id=1"""
    dato = {
        "tiempo": str(datetime.utcnow()),
        "id": 1,
        "bateria": random.randint(0, 100),
        "velocidad": random.randint(0, 5),
        "punto": str(Punto(random.randint(0, 5), random.randint(0,5))),
        "siguiente_punto": str(Punto(random.randint(0, 5), random.randint(0,5))),
        "pos_x": random.random() * 5,
        "pos_y": random.random() * 5,

    }

    return json.dumps(dato)

def simular_csv(socket, csv_path):
    time = datetime.utcnow()
    with open("/simulator/" + csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            delta = timedelta(seconds=float(row['time']))
            while datetime.utcnow() < time + delta:
                pass
            row['time'] = (time + delta).strftime('%Y-%m-%d %H:%M:%S.%f')
            sock.sendto(bytes(json.dumps(row), encoding="utf-8"), (UDP_IP, UDP_PORT))


def simular_agv(socket):
    puntos = [
        Punto(0, 0),
        Punto(1, 0),
        Punto(0, 1),
        Punto(1, 1)
    ]
    mapa = Mapa(puntos)
    agv = AGV(puntos[0], puntos[1])

    agv.simular(sock)

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    f = open('/simulator/config.json')
    data = json.load(f)

    if data['from_csv']:
        simular_csv(sock, data['csv_file'])
    else:
        simular_agv(sock)