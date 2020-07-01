_Author_ = "Karthik Vaidhyanathan"

# Script to perform reactive adaptation based on the algorithm


import csv
import sys
import time
from kafka import KafkaConsumer, KafkaProducer
from Initializer import Initialize
from Custom_Logger import logger
import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib

import tensorflow as tf

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K

from Adaptation_Planner import Adaptation_Planner

from Initializer import Initialize


init_obj = Initialize()
ada_obj = Adaptation_Planner()

import json
from Initializer import Initialize

prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
             19160.0]  # Initialize the inital energy configuration

#energy_model_file_json = "model_lstm_energy2_v2_ colab.json"
energy_model_file_json = init_obj.adaptation_model_json
#energy_model_file_h5 = "model_lstm_energy_v2_colab.h5"
#energy_model_file_h5 = "model_lstm_energy_v3_H10_colab.h5"
energy_model_file_h5 = init_obj.adaptation_model_h5

# Load the Machine learning models

#### scalars #####
scalar_energy = joblib.load(init_obj.scalar_path + init_obj.adaptation_model_scalar)

graph = tf.get_default_graph()


json_file_energy = open(init_obj.model_path + energy_model_file_json, 'r')
loaded_model_energy_json = json_file_energy.read()
json_file_energy.close()
loaded_model_energy = model_from_json(loaded_model_energy_json)
# load weights into new model
loaded_model_energy.load_weights(init_obj.model_path + energy_model_file_h5)
print("Loaded model from disk")

main_energy_list = []
class Streaming_Consumer():
    # Class that will perform the prediction in near-real time
    def process_sensor_data(self):
        # This will process the data from the sensor and then perform the management of the data
        print ("processing")
    def gather_data(self,adaptation_type,horizon=10,lag=10,decision_period=10):
        global prev_vals
        global main_energy_forecast
        global main_traffic_forecast
        consumer = KafkaConsumer(auto_offset_reset='latest',
                                  bootstrap_servers=['localhost:9092'], api_version=(0, 10), consumer_timeout_ms=1000)

        consumer.subscribe(pattern='^sensor.*')    # Subscribe to a pattern
        main_energy_list = []
        while True:
            for message in consumer:
                if message.topic == "sensor":
                    # The QoS data comes here and the prediction needs to be done here
                    row = str(message.value).split(";")
                    if (len(row) > 3):
                        time_string = row[0]
                        second_level_data = []
                        row.pop()  # remove the unwanted last element
                        vals = [x1 - float(x2) for (x1, x2) in zip(prev_vals, row[1:])]
                        # print (len (vals))
                        if (len(vals) == 22):
                            # Check if we have 22 elements always
                            # spark_predictor.main_energy_list.append(vals)
                            main_energy_list.append(vals)
                            #final_energy_list = [x + y for x, y in zip(final_energy_list, vals)] ## Keep addding them
                            prev_vals = [float(i) for i in row[1:]]

                    if adaptation_type == "reactive":
                        if (len(main_energy_list) == 1):
                            #print (main_energy_list)
                            # Compute the energy consumed by each sensor
                            ada_obj.reactive(main_energy_list)
                            logger.info("adaptation count " + str(ada_obj.adapation_count) + " " + str(ada_obj.time_count))
                            main_energy_list = [] # This will mean only every 10 minutes an adaptation will be performed
                    elif adaptation_type == "proactive":
                        #print (ada_obj.adapation_count)
                        if (len(main_energy_list) == lag):
                            print ("reached")
                            predict_array = np.array(main_energy_list)
                            predict_array = scalar_energy.fit_transform(predict_array)
                            # print (predict_array.shape)
                            predict_array = predict_array.reshape(1, lag, 22)
                            with graph.as_default():
                                energy_forecast = loaded_model_energy.predict(predict_array)
                            # K.clear_session()
                            inverse_forecast = energy_forecast.reshape(horizon, 22)
                            inverse_forecast = scalar_energy.inverse_transform(inverse_forecast)
                            inverse_forecast_features = inverse_forecast.reshape(energy_forecast.shape[0], 22*horizon)
                            energy_forecast_total = 0
                            for j in range(0, inverse_forecast_features.shape[1]):
                            #for j in range(0, 22*horizon): # Number of components * horizon equals inverse_forecast_Features.shape[1]
                                if j not in [20,42,64,86,108,130,152,174,196,218,240,262,284,306,328,350,372,394,416,438,460,482,504,526,548,570,592,614,636,658]:
                                    # Ignore the database forecasts
                                    energy_forecast_total = energy_forecast_total + inverse_forecast_features[0, j]

                            #print("Energy forecast")
                            #print(energy_forecast_total)
                            ada_obj.proactive(inverse_forecast_features,energy_forecast_total,horizon=horizon)
                            logger.info("adaptation count " + str(ada_obj.adapation_count) + " " + str(ada_obj.time_count))
                            #main_energy_list = []  # This will mean only every 10 minutes an adaptation will be performed
                            main_energy_list = main_energy_list[decision_period:]  # This will mean only every 10 minutes an adaptation will be performed


if __name__ == '__main__':
    stream_consumer =  Streaming_Consumer()
    stream_consumer.gather_data(adaptation_type=init_obj.adaptation_type,horizon=init_obj.horizon,lag=init_obj.lag,decision_period=init_obj.decision_period) # adaptation_type denotes the type of adaptation to be performed





