_Author_ = "*******************"

# Program to consume and perform the prediction of the data as and when received

import csv
import sys
import time
from kafka import KafkaConsumer, KafkaProducer
from Initializer import Initialize
from Custom_Logger import logger
import numpy as np
from py4j.java_gateway import JavaGateway     # For calling the model checker from Python
from numpy import array
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib

from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
import Spark_Predictor_RLMC
#import Spark_Predicto  ## Without Model checking


import json

#from Spark_Predictor_RLMC import Spark_Predictor # To run with RL and MC
#from Spark_Predictor import Spark_Predictor  # To run just with RL
from Spark_Predictor_MC import Spark_Predictor # To run with MC
from Initializer import Initialize

# start_time = datetime.now()
prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
             19160.0]  # Initialize the inital energy configuration

# last_read = 0
# sql_connector = sqlUtils.SqlConnector()   # SQL util class

# es = Elasticsearch() # Connect to the default ES
#   init_object = Initialize()
spark_predictor_obj = Spark_Predictor()
current_pattern = "co"

init_object = Initialize()

model_path =  init_object.model_path
energy_model_file_h5 = "model1_master_29Nov.h5"
energy_model_file_json = "model1_master_29Mov.json"
traffic_model_file_h5 = "model_traffic_ada_29Nov.h5"
traffic_model_file_json = "model_traffic_ada_29Nov.json"

#### scalars #####
scalar_energy = joblib.load("scaler.save")
scalar_traffic = joblib.load("scaler_traffic_CO.save")

graph = tf.get_default_graph()
main_list = []
main_list_traffic = []

json_file_energy = open(model_path + energy_model_file_json, 'r')
loaded_model_energy_json = json_file_energy.read()
json_file_energy.close()
loaded_model_energy = model_from_json(loaded_model_energy_json)
# load weights into new model
loaded_model_energy.load_weights(model_path + energy_model_file_h5)
print("Loaded model from disk")

json_file_traffic = open(model_path + traffic_model_file_json, 'r')
loaded_model_traffic_json = json_file_traffic.read()
json_file_traffic.close()
loaded_model_traffic = model_from_json(loaded_model_traffic_json)
# load weights into new model
loaded_model_traffic.load_weights(model_path + traffic_model_file_h5)
print("Loaded model from disk")
# K.clear_session()

gateway = JavaGateway()
analyze = gateway.entry_point.getAnalyze()
main_energy_forecast = 0
main_traffic_forecast = 0

# Create the sensor dict
sensor_list = ["S34", "S33", "S24", "S25", "S41", "S42", "S1", "S2", "S18", "S20"]

sensor_model_dict = {}
sensor_scalar_dict = {}
for sensor_id in sensor_list:
    model_path = "/Users/karthikvaidhyanathan/Dropbox/PyCharmProjects/ArchLearner/test_models/"
    json_file = open(model_path + "model_mode_" + sensor_id + ".json", 'r')
    loaded_model_sensor_json = json_file.read()
    json_file.close()
    loaded_model_sensor = model_from_json(loaded_model_sensor_json)
    # load weights into new model
    loaded_model_sensor.load_weights(model_path + "model_mode_" + sensor_id + ".h5")
    print("Loaded model from disk")
    sensor_model_dict[sensor_id] = loaded_model_sensor
    sensor_scalar_dict[sensor_id] = joblib.load(model_path + "scaler_mode" + sensor_id + ".save")

with open(init_object.json_path + "archlearner_spark_output_su.json") as json_file:
    output_json_su = json.load(json_file)

with open(init_object.json_path + "archlearner_spark_output_sc.json") as json_file:
    output_json_sc = json.load(json_file)

with open(init_object.json_path + "archlearner_spark_output_co.json") as json_file:
    output_json_co = json.load(json_file)

# Set up the sensor dictionary
sensor_list = ["S34", "S33", "S24", "S25", "S41", "S42", "S1", "S2", "S18", "S20"]
sensor_dict = {}
for key in sensor_list:
    sensor_dict[key] = []

def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    # print(inverted)
    return inverted


def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        # print (forecast)
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # print (inv_scale)
        # invert differencing
        # print ("length " ,len(series))
        index = len(series) - n_test + i - 1
        # print (index)
        last_ob = series.values[index]

        # print (last_ob)
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


def predict_sensor_mode(sensor_id):
    # Takes the sensor id and performs the mode forecast

    predict_array = np.array(sensor_dict[sensor_id])
    predict_array = predict_array.reshape(1, 10, 1)
    with graph.as_default():
        data_forecasts = sensor_model_dict[key].predict(predict_array)
    times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Corresponding to the last 10 minute data
    #current_df_data = pd.DataFrame(sensor_dict[sensor_id], index=times)
    # print (key)
    #data_forecasts = inverse_transform(current_df_data, data_forecasts, sensor_scalar_dict[sensor_id], n_test=10)
    data_forecasts = sensor_scalar_dict[sensor_id].inverse_transform(data_forecasts)
    # predicted_values = [[9] for traffic_forecast in traffic_forecasts]
    forecasted_modes = {}  # to prepare the forecast of modes for each sensor
    normal_limit = output_json_co["sensors"][sensor_id]["modes"][
        "normal_limit"]  # Get the normal limit from the configurations
    normal_freq = output_json_co["sensors"][sensor_id]["modes"][
        "normal_freq"]
    critical_freq = output_json_co["sensors"][sensor_id]["modes"][
        "critical_freq"]
    time_start = 20
    print ("Data forecasts")
    print (data_forecasts)
    #for data in data_forecasts:
    count = 0
    for values in data_forecasts[0]:
        count += 1
        if values<=0:
            values = values*-1
        if values <= normal_limit:
            # start_time = time_start
            while (time_start / 60 < count):
                time_start = time_start + normal_freq
                if (time_start > 600):  # Ensure that it does not cross the limit
                    time_start = 600
                forecasted_modes[time_start] = 1
        elif values > normal_limit:
            while (time_start / 60 < count):
                time_start = time_start + critical_freq
                if (time_start > 600):  # Ensure that it does not cross the limit
                    time_start = 600
                forecasted_modes[time_start] = 2

    # print (forecasted_modes)
    top_key = next(iter(forecasted_modes))
    prev_val = current_val = forecasted_modes[top_key]
    prev_key = current_key = 0
    forecasted_modes_final = []
    # forecasted_modes_final[str(prev_key) + "_" + str(top_key)] = current_val # Insert the first element
    # prev_key = top_key

    for key_val in forecasted_modes.keys():
        key_dict = {}
        current_val = forecasted_modes[key_val]
        if current_val != prev_val:
            # forecasted_modes_final[str(prev_key) + "_" + str(key_val)] = prev_val
            if prev_val == 1:
                key_dict["start"] = int(float(prev_key))
                key_dict["end"] = int(float(key_val))
                key_dict["mode"] = "normal"
            elif prev_val == 2:
                key_dict["start"] = int(float(prev_key))
                key_dict["end"] = int(float(key_val))
                key_dict["mode"] = "critical"
            forecasted_modes_final.append(key_dict)
            #print(key_dict)
            prev_val = current_val
            prev_key = key_val

    key_dict = {}
    # forecasted_modes_final[str(prev_key) + "_" + str(key_val)] = prev_val

    if prev_val == 1:
        key_dict["start"] = int(float(prev_key))
        key_dict["end"] = int(float(key_val))
        key_dict["mode"] = "normal"
    elif prev_val == 2:
        key_dict["start"] = int(float(prev_key))
        key_dict["end"] = int(float(key_val))
        key_dict["mode"] = "critical"
    forecasted_modes_final.append(key_dict)

    print(forecasted_modes_final)
    return forecasted_modes_final

class Streaming_Consumer():
    # Class that will perform the prediction in near-real time
    def process_sensor_data(self):
        # This will process the data from the sensor and then perform the management of the data
        print ("processing")
    def gather_data(self):
        global prev_vals
        global main_energy_forecast
        global main_traffic_forecast
        consumer = KafkaConsumer(auto_offset_reset='latest',
                                  bootstrap_servers=['localhost:9092'], api_version=(0, 10), consumer_timeout_ms=1000)

        consumer.subscribe(pattern='^sensor.*')    # Subscribe to a pattern
        main_list = []
        main_list_traffic = []
        while True:
            for message in consumer:
                #print (message.topic)
                if message.topic == "sensorData":
                    string_val = str(message.value)
                    string_val = string_val.strip("b'").strip("\n")
                    row = string_val.split(";")
                    sensor_id = row[0]
                    data = float(row[2])
                    sensor_dict[sensor_id].append(data)
                    if len(sensor_dict[sensor_id]) == 10:
                        forecast_modes = predict_sensor_mode(sensor_id)
                        output_json_co["sensors"][key]["forecasted_modes"] = forecast_modes  ## A List containing the forecasts
                        output_json_su["sensors"][key]["forecasted_modes"] = forecast_modes  ## A List containing the forecasts
                        output_json_sc["sensors"][key]["forecasted_modes"] = forecast_modes  ## A List containing the forecasts
                        #print(output_json["sensors"][key]["forecasted_modes"])
                        sensor_dict[sensor_id] = []  # Reinitalize to make it 0 again

                    # Save the JSON file as and when needed for all three JSONS
                        with open("./jsons/archlearner_spark_output_"+ "co" +".json", "w") as json_file:
                            json.dump(output_json_co, json_file)
                        with open("./jsons/archlearner_spark_output_" + "su" + ".json", "w") as json_file:
                            json.dump(output_json_su, json_file)
                        with open("./jsons/archlearner_spark_output_" +"sc" + ".json", "w") as json_file:
                            json.dump(output_json_sc, json_file)
                        #print (json.dumps(output_json))
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
                            main_list.append(vals)
                            prev_vals = [float(i) for i in row[1:]]

                    elif (len(row) == 2):
                        # This is the case for data traffic
                        # spark_predictor.main_data_traffic_list.append(row[1])
                        row[1] = row[1].strip("'")
                        main_list_traffic.append(float(row[1]))

                    energy_forecast_total = 0
                    data_traffic_forecast = 0
                    flag = 0
                    flag2 =0
                    if (len(main_list) == 10):
                        print (main_list)
                        predict_array = np.array(main_list)
                        # print (predict_array.shape)
                        predict_array = predict_array.reshape(1, 10, 22)
                        with graph.as_default():
                            energy_forecast = loaded_model_energy.predict(predict_array)
                        # K.clear_session()
                        inverse_forecast = energy_forecast.reshape(10, 22)
                        inverse_forecast = scalar_energy.inverse_transform(inverse_forecast)
                        # print (inverse_forecast)

                        inverse_forecast_features = inverse_forecast.reshape(energy_forecast.shape[0], 220)
                        for j in range(0, inverse_forecast_features.shape[1]):
                            if j not in [1, 23, 45, 67, 89, 111, 133, 155, 177, 199, 15,37,59,81,103,125,147,169,191,213,16,38,60,82,104
                                         ,126,148,170,193,214,17,39,61,83,105
                                            ,127,149,171,194,215,18,40,62,84,106
                                                ,128,150,172,195,216, 19,41,63,85,107
                                                ,129,151,173,196,217, 20,42,64,86,108
                                                ,130,152,174,197,218]:
                                energy_forecast_total = energy_forecast_total + inverse_forecast_features[0, j]

                        print ("Energy forecast")
                        print (energy_forecast_total)
                        main_energy_forecast =  energy_forecast_total
                        flag = 1
                        # print(main_list)
                        main_list = []
                        #main_list.pop(0)

                    if (len(main_list_traffic) == 10):
                        predict_array_traffic = np.array(main_list_traffic)
                        predict_array_traffic = predict_array_traffic.reshape(1, 10, 1)
                        with graph.as_default():
                            traffic_forecasts = loaded_model_traffic.predict(predict_array_traffic)
                        # K.clear_session()
                        # scalar_traffic = joblib.load("scaler.save")
                        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                        current_df_traffic = pd.DataFrame(main_list_traffic, index=times)
                        traffic_forecasts = inverse_transform(current_df_traffic, traffic_forecasts, scalar_traffic,
                                                              n_test=10)
                        predicted_traffic = [traffic_forecast[9] for traffic_forecast in traffic_forecasts]

                        data_traffic_forecast = predicted_traffic[0][0]
                        print ("data forecast")
                        print (data_traffic_forecast)
                        main_traffic_forecast = data_traffic_forecast
                        # remove first element
                        #main_list_traffic.pop(0)
                        main_list_traffic = []
                        flag = 1
                    if flag==1:
                        if (main_energy_forecast>0 and main_traffic_forecast >0):
                            print("calling reinforcer")
                            # print ("len " + str(len(main_list_traffic)))
                            # print (data_traffic_forecast)
                            spark_predictor_obj.predictor(analyze,main_energy_forecast, main_traffic_forecast)
                            flag = 0
                            main_traffic_forecast = 0
                            main_energy_forecast = 0
                        #print (len(main_list_traffic))
                        #print (len(main_list))

if __name__ == '__main__':
    stream_consumer =  Streaming_Consumer()
    stream_consumer.gather_data()

