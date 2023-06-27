FROM python:3.10-slim

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /forecasting
COPY . ./

CMD ["python", "-u", "src/main.py"]
