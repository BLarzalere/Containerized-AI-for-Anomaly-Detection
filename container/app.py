"""
This python file uses the Flask framework to accept sensor data via a REST API for anomaly detection  
by an AI neural network. The neural network model has been pre-trained is loaded and executed 
using Keras and TensorFlow.

Usage:
Start the server:
   python app.py
Submit a request via cURL:
   curl -X POST -F data_file=@day4_data.csv 'http://localhost:5000/submit'

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import flask
from keras.models import load_model
from sklearn.externals import joblib
import tensorflow as tf
import csv
import codecs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# initialize the Flask application
app = flask.Flask(__name__)


# load the pre-trained Keras model
def define_model():
    global graph
    graph = tf.compat.v1.get_default_graph()
    global model
    model = load_model('Cloud_model.h5')
    return print("Model Loaded")


# define anomaly threshold
limit = 0.275


# this method processes any requests to the /submit endpoint
@app.route("/submit", methods=["POST"])
def submit():
    # initialize the data dictionary that will be returned in the response
    data_out = {}

    # load the data file from our endpoint
    if flask.request.method == "POST":

        # read the data file
        file = flask.request.files["data_file"]
        if not file:
            return "No file submitted"
        data = []
        stream = codecs.iterdecode(file.stream, 'utf-8')
        for row in csv.reader(stream, dialect=csv.excel):
            if row:
                data.append(row)

        # convert input data to pandas dataframe
        df = pd.DataFrame(data)
        df.set_index(df.iloc[:, 0], inplace=True)
        df2 = df.drop(df.columns[0], axis=1)
        df2 = df2.astype(np.float64)

        # normalize the data
        scaler = joblib.load("./scaler_data")
        X = scaler.transform(df2)
        # reshape data set for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        # calculate the reconstruction loss on the input data
        with graph.as_default():
            data_out["Analysis"] = []
            preds = model.predict(X)
            preds = preds.reshape(preds.shape[0], preds.shape[2])
            preds = pd.DataFrame(preds, columns=df2.columns)
            preds.index = df2.index

            scored = pd.DataFrame(index=df2.index)
            yhat = X.reshape(X.shape[0], X.shape[2])
            scored['Loss_mae'] = np.mean(np.abs(yhat - preds), axis=1)
            scored['Threshold'] = limit
            scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

            # determine if an anomaly was detected
            triggered = []
            for i in range(len(scored)):
                temp = scored.iloc[i]
                if temp.iloc[2]:
                    triggered.append(temp)
            print(len(triggered))
            if len(triggered) > 0:
                for j in range(len(triggered)):
                    out = triggered[j]
                    result = {"Anomaly": True, "value": out[0], "filename": out.name}
                    data_out["Analysis"].append(result)

                # comment out for containerized deployment 
                # baseline = pd.read_pickle("baseline.pkl")
                # combined = pd.concat([baseline, scored])
                # combined.plot(logy=True, figsize=(16, 9), ylim=[1e-2, 1e2], color=['blue', 'red'])
                # plt.savefig('anomaly_plot.png')

            else:
                result = {"Anomaly": "No Anomalies Detected"}
                data_out["Analysis"].append(result)

    # return the data dictionary as a JSON response
    return flask.jsonify(data_out)


# first load the model and then start the server
# we need to specify the host of 0.0.0.0 so that the app is available on both localhost as well
# as on the external IP of the Docker container
if __name__ == "__main__":
    print(("* Loading the Keras model and starting the server..."
          "Please wait until the server has fully started before submitting"))
    define_model()
    app.run(host='0.0.0.0')
