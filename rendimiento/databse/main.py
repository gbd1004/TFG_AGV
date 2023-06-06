import docker
import json
import time
import threading
from datetime import datetime

from collections import OrderedDict
from csv import DictReader

import dateutil.parser

import reactivex as rx
from reactivex import operators as ops

from influxdb_client import Point, InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

import psycopg2

from prometheus_client import Gauge, start_http_server
from prometheus_client.core import GaugeMetricFamily

client = docker.from_env()

# START CONTAINERS

def start_influx_container():
    container = client.containers.run("influxdb:2.6.1", ports={'8086/tcp': 8086}, detach=True,
                                      auto_remove=True)
    time.sleep(1)
    container.exec_run(cmd="influx setup -u gonzalo -p prueba1234 -t 123456789 -o UBU -b TFG -f")
    return container

def start_timescaledb_container():
    container = client.containers.run("timescale/timescaledb:latest-pg15", ports={'5432/tcp': 5432}, detach=True,
                                      auto_remove=True, environment=["POSTGRES_PASSWORD=password"])
    time.sleep(1)
    return container

def start_prometheus_container():
    container = client.containers.run("prom/prometheus:latest", ports={'9090/tcp': 9090}, detach=True,
                                      auto_remove=True, volumes={'/home/gonzalo/repos/TFG_AGV/rendimiento/prometheus.yml': {'bind': '/etc/prometheus/prometheus.yml', 'mode': 'rw'}},
                                      extra_hosts={"host.docker.internal":"host-gateway"})
                                    #   auto_remove=True, volumes={'/home/gonzalo/repos/TFG_AGV/rendimiento/prometheus.yml': {'bind': '/etc/prometheus/prometheus.yml', 'mode': 'rw'}})
    time.sleep(1)
    return container

# TEST FUNCTIONS

def write_test(write_func, id):
    stop = False
    x = threading.Thread(target=get_stats_info, args=(id, lambda : stop))

    x.start()

    elapsed = write_func()

    stop = True

    print("Tiempo insercion: ", elapsed, " s")
    print("Inserciones/segundo: ", 300000 / elapsed)
    x.join()

def tiempo_respuesta_test(target_insert, target_query):
    global insert_time
    global query_time

    dif = 0

    for i in range(0, 200):
        insert_thread = threading.Thread(target=target_insert)
        query_thread = threading.Thread(target=target_query)
        insert_thread.start()
        query_thread.start()

        query_thread.join()
        insert_thread.join()

        dif += query_time - insert_time
        print("\r", (i / 200) * 100, end="")
        time.sleep(0.300)
    dif /= 200
    print()
    print(dif / 1_000_000, "ms")

# INFLUXDB RELATED FUNCTIONS

def parse_row(row: OrderedDict):
    return Point("datos_agv") \
        .tag("id", str(row['id'])) \
        .field("bateria", float(row['bateria'])) \
        .field("velocidad", float(row['velocidad'])) \
        .field("punto_x", float(row['punto_x'])) \
        .field("punto_y", float(row['punto_y'])) \
        .field("temperatura", float(row['temperatura'])) \
        .field("voltaje", float(row['voltaje'])) \
        .time(row['tiempo'])

data = rx \
    .from_iterable(DictReader(open('datos.csv', 'r'))) \
    .pipe(ops.map(lambda row: parse_row(row)))

def write_influx():
    start = time.perf_counter()
    with InfluxDBClient(url="http://localhost:8086", token="123456789", org="UBU", debug=False) as client:
        with client.write_api(write_options=WriteOptions(batch_size=5_000, flush_interval=10_000)) as write_api:
            write_api.write(bucket="TFG", record=data)
    elapsed = time.perf_counter() - start
    return elapsed

insert_time = 0
query_time = 0

def insert_influx():
    with InfluxDBClient(url="http://localhost:8086", token="123456789", org="UBU", debug=False) as client:
        global insert_time
        time.sleep(1)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        data = Point("datos_agv") \
                    .tag("id", "5") \
                    .field("bateria", 50.2) \
                    .field("velocidad", 10.0) \
                    .field("punto_x", 1.0) \
                    .field("punto_y", 2.0) \
                    .field("temperatura", 23.5) \
                    .field("voltaje", 1.5) \
                    .time(datetime.utcnow())
        write_api.write(bucket="TFG", record=data)
        insert_time = time.perf_counter_ns()

def query_influx():
    with InfluxDBClient(url="http://localhost:8086", token="123456789", org="UBU", debug=False) as client:
        global query_time
        query = '''from(bucket: "TFG") 
                    |> range(start: -200ms) 
                    |> filter(fn: (r) => r["_measurement"] == "datos_agv") 
                    |> filter(fn: (r) => r["_field"] == "bateria") 
                    |> filter(fn: (r) => r["id"] == "5")'''
        query_api = client.query_api()
        
        while True:    
            result = query_api.query(query=query)
            query_time = time.perf_counter_ns()

            if len(result) != 0:
                break

# TIMESCALEDB RELATED FUNCTIONS

TSDB_CONN = "postgres://postgres:password@localhost:5432/postgres"
def write_tsdb():
    start = time.perf_counter()

    query_create_table = """CREATE TABLE datos_agv(
                                time TIMESTAMPTZ NOT NULL,
                                id INTEGER,
                                bateria DECIMAL,
                                velocidad DECIMAL,
                                punto_x DECIMAL,
                                punto_y DECIMAL,
                                temperatura DECIMAL,
                                voltaje DECIMAL
                            );"""
    query_create_hypertable = "SELECT create_hypertable('datos_agv', 'time');"

    with psycopg2.connect(TSDB_CONN) as conn:
        cursor = conn.cursor()
        cursor.execute(query_create_table)
        cursor.execute(query_create_hypertable)
        conn.commit()
        cursor.close()

    query = "COPY datos_agv FROM STDIN DELIMITER ',' CSV HEADER;"
    csv_file = 'datos.csv'

    with psycopg2.connect(TSDB_CONN) as conn:
        cursor = conn.cursor()
        cursor.copy_expert(sql=query, file=open(csv_file, "r"), size=5000)
        conn.commit()
        cursor.close()

    elapsed = time.perf_counter() - start
    return elapsed

def insert_tsdb():
    query = """INSERT INTO datos_agv(time, id, bateria, velocidad, punto_x, punto_y, temperatura, voltaje) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"""
    time_query = datetime.utcnow()

    with psycopg2.connect(TSDB_CONN) as conn:
        global insert_time
        cursor = conn.cursor()
        time.sleep(1)
        cursor.execute(query, (time_query, 5, 50.2, 10.0, 1.0, 2.0, 23.5, 1.5))
        conn.commit()
        insert_time = time.perf_counter_ns()
        cursor.close()

def query_tsdb():
    query = """SELECT * FROM datos_agv WHERE time > now() - INTERVAL '200 milliseconds';"""

    with psycopg2.connect(TSDB_CONN) as conn:
        global query_time
        cursor = conn.cursor()

        while True:
            cursor.execute(query)
            result = cursor.fetchall()
            query_time = time.perf_counter_ns()

            if len(result) != 0:
                break

        cursor.close()

# PROMETHEUS RELATED FUNCTIONS

def write_prom():
    start = time.perf_counter()

    bateria = Gauge("nivel_bateria", "nivel de la batería", ['agv_id'])
    velocidad = Gauge("velocidad", "Velocidad del vehiculo", ['agv_id'])
    punto_x = Gauge("punto_x", "Coordenada x del vehiculo", ['agv_id'])
    punto_y = Gauge("punto_y", "Coordenada y del vehiculo", ['agv_id'])
    temperatura = Gauge("temperatura", "Temperatura de la bateria", ['agv_id'])
    voltaje = Gauge("voltaje", "voltaje de la bateria", ['agv_id'])

    with open('datos.csv', newline='') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            bateria.labels(row['id']).set(float(row['bateria']))
            velocidad.labels(row['id']).set(float(row['velocidad']))
            punto_x.labels(row['id']).set(float(row['punto_x']))
            punto_y.labels(row['id']).set(float(row['punto_y']))
            temperatura.labels(row['id']).set(float(row['temperatura']))
            voltaje.labels(row['id']).set(float(row['voltaje']))
            time.sleep(0.000001)

    elapsed = time.perf_counter() - start
    return elapsed

# STATS RELATED FUNCTIONS

def get_cpu_stats(stats):
    UsageDelta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']

    SystemDelta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']

    len_cpu = stats['cpu_stats']['online_cpus']

    percentage = (UsageDelta / SystemDelta) * len_cpu * 100

    return percentage

def get_mem_stats(id = 0, with_stats = False, stats_arg = None):
    if not with_stats:
        stats = client.containers.get(id).stats(stream=False)
    else:
        stats = stats_arg

    memory = stats['memory_stats']['usage']
    return memory

def get_stats_info(id, stop):
    uso_cpu = []
    uso_memoria = []
    while True:
        stats = client.containers.get(id).stats(stream=False)

        uso_cpu.append(get_cpu_stats(stats))
        uso_memoria.append(get_mem_stats(with_stats=True, stats_arg=stats))

        if stop():
            print("Uso medio de CPU: ", sum(uso_cpu) / len(uso_cpu), " %")
            print("Uso maximo de memoria: ", max(uso_memoria), " B")
            break
def main():
    container = start_influx_container()
    id = container.attrs['Id']

    print("INFLUXDB")
    print("TEST INSERCION 300.000 PUNTOS")
    time.sleep(5)
    write_test(write_func=write_influx, id=id)

    print("\nTEST TIEMPO DE RESPUESTA ENTRE INSERT Y SELECT")
    tiempo_respuesta_test(target_insert=insert_influx, target_query=query_influx)

    time.sleep(3)
    container.stop()

    container = start_timescaledb_container()
    id = container.attrs['Id']

    print("\n\nTIMESCALEDB")
    print("TEST INSERCION 300.000 PUNTOS")
    time.sleep(5)
    write_test(write_func=write_tsdb, id=id)

    print("\nTEST TIEMPO DE RESPUESTA ENTRE INSERT Y SELECT")
    tiempo_respuesta_test(target_insert=insert_tsdb, target_query=query_tsdb)

    time.sleep(3)
    container.stop()

if __name__ == "__main__":
    main()