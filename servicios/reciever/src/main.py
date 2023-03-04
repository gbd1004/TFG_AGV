import json
import socket
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

UDP_IP = "0.0.0.0"
UDP_PORT = 5004
BUCKET = "AGV"
ORG = "TFG"
TOKEN = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
URL = "http://database:8086"

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    client = influxdb_client.InfluxDBClient(
        url=URL,
        token=TOKEN,
        org=ORG
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    while True:
        data, addr = sock.recvfrom(1024)       
        data_json = json.loads(data)

        point = influxdb_client.Point("agv_logs") \
                    .tag("vid", str(data_json["id"])) \
                    .field("bateria", int(data_json["bateria"])) \
                    .field("velocidad", int(data_json["velocidad"])) \
                    .field("estacion", int(data_json["estacion"])) \
                    .time(data_json["tiempo"])
        
        write_api.write(bucket=BUCKET, org=ORG, record=point)