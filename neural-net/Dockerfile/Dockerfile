FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt /requirements.txt

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip \
    && apt-get install ffmpeg -y libsm6 libxext6

RUN pip3 install -r /requirements.txt