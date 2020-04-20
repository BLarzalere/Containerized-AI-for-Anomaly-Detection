FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY app.py /app
COPY Cloud_model.h5 /app
COPY scaler_data /app

CMD ["python", "app.py"]
