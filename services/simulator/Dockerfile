FROM python:3.10-slim

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /simulator
COPY . ./

EXPOSE 5005/udp

CMD ["python", "-u", "src/main.py"]
