FROM python:3.8.5
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY app.py /app
COPY Cloud_model.h5 /app
COPY scaler_data /app

#ENV FLASK_DEBUG=1

CMD ["python", "app.py"]
