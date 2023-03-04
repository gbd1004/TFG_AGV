import json
import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 5004

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    while True:
        data, addr = sock.recvfrom(1024)
       
        data_json = json.loads(data)
        print(data_json)
        # print(data.decode("utf-8"))
