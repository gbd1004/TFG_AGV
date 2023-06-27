FROM python:3.10-slim

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /reciever
COPY . ./

EXPOSE 5004/udp

CMD ["python", "-u", "src/main.py"]
