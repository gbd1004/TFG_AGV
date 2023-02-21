#!/bin/bash

if [[ "$(sudo docker container inspect influxdb 2> /dev/null)" == "[]" ]]; then
    sudo docker run --name mongo -d -p 27017:27017 mongo

    # sleep 5

    # sudo docker exec -it -d influxdb influx setup -u gonzalo -p prueba1234 -t 123456789 -o UBU -b TFG -f
else
    sudo docker start mongo
fi