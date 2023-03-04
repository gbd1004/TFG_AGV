"""Servicio dedicado a simular el AGV. De momento solo son datos aleatorios"""

import socket
import random
import json
import sys
import time

UDP_IP = "reciever"
UDP_PORT = 5004

def generar_datos():
    """Genera datos aleatorios del AGV con id=1"""
    dato = {
        "tiempo": time.time(),
        "id": 1,
        "bateria": random.randint(0, 100),
        "velocidad": random.randint(0, 5),
        "estacion": random.randint(0, 10)

    }

    return json.dumps(dato)

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        datos = generar_datos()
        sock.sendto(bytes(datos, encoding="utf-8"), (UDP_IP, UDP_PORT))
        print(datos)

        time.sleep(5)
