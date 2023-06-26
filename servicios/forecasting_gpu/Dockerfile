FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /forecasting_gpu
COPY . ./

RUN ls /

CMD ["python3", "-u", "src/main.py"]