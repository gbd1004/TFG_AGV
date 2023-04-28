import graphyte
import time
import random

graphyte.init('localhost', prefix='system.sync')
while True:
    a = random.random() * 10
    print(a)
    graphyte.send('prueba.bar', a)
    time.sleep(1)