FROM python:3

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /forecasting_cpu
COPY . ./

CMD ["python", "-u", "src/main.py"]
