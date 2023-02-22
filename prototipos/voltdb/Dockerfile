FROM ubuntu:latest

RUN apt-get -y update --allow-unauthenticated

RUN apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get -y install git python26 openjdk-8-jdk ant build-essential ant-optional \
    cmake ccache apt-show-versions ntp

RUN git clone https://github.com/VoltDB/voltdb.git && cd voltdb

RUN cd voltdb && ant