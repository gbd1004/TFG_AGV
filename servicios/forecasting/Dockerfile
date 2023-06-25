FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04
RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /forecasting
COPY . ./

CMD ["python3", "-u", "src/main.py"]
