import csv
from datetime import datetime
import random

with open('datos.csv', mode='w') as csvfile:
    fieldnames = ['tiempo', 'id', 'bateria', 'velocidad', 'punto_x', 'punto_y', 'temperatura', 'voltaje']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(100000):
        for i in range(1, 4):
            dato = {
                "tiempo": datetime.utcnow(),
                "id": i,
                "bateria": random.randint(0, 100),
                "velocidad": random.randint(0, 5),
                "punto_x": random.randint(0, 5),
                "punto_y": random.randint(0,5),
                "temperatura": random.randint(20, 40),
                "voltaje": random.random() * 2,
            }

            writer.writerow(dato)