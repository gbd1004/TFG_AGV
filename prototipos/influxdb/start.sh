#!/bin/bash

if [[ "$(sudo docker container inspect influxdb 2> /dev/null)" == "[]" ]]; then
    sudo docker run -it -d --name influxdb -p 8086:8086 influxdb:2.6.1

    sleep 5

    sudo docker exec -it -d influxdb influx setup -u gonzalo -p prueba1234 -t 123456789 -o UBU -b TFG -f
else
    sudo docker start influxdb
fi