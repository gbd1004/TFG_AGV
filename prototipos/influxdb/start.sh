#!/bin/bash

if [[ "$(sudo docker images -q influxdb:2.6.1 2> /dev/null)" == "" ]]; then
    sudo docker start influxdb
else
    sudo docker run --name influxdb -p 8086:8086 influxdb:2.6.1
fi