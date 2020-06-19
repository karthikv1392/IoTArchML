_Author_ = "Karthik vaidhyanathan"


import pandas as pd
from Initializer import Initialize

import numpy as np
import time
import random
import os
import csv
import sys
import signal
import keras
from keras.models import model_from_json
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


init_object = Initialize()



def equalize_indexes():
    # Read the data from csv files and return dataframes after equalizing the indexes

    df_high = pd.read_csv(init_object.data_path + "aggregated_data_high.csv",
                          sep=",",
                          index_col="timestamp")  # Read the proccessed data frame

    df_medium = pd.read_csv(init_object.data_path + "aggregated_data_medium.csv",
                            sep=",",
                            index_col="timestamp")  # Read the proccessed data frame

    df_low = pd.read_csv(init_object.data_path + "aggregated_data_low.csv",
                         sep=",",
                         index_col="timestamp")  # Read the proccessed data frame

    df_medium.index = df_high.index
    df_low.index = df_high.index

    plt.plot(df_high["S1"])
    plt.plot(df_medium["S1"])
    plt.plot(df_low["S1"])
    plt.savefig("sample.png")

    print (df_medium)

    print (df_low)
    return df_high,df_medium,df_low

def load_model():

    # Load the model

    json_file_co = open(init_object.model_path + 'model1.json', 'r')
    loaded_model_co_json = json_file_co.read()
    json_file_co.close()
    loaded_model_co = model_from_json(loaded_model_co_json)
    # load weights into new model
    loaded_model_co.load_weights(init_object.model_path + "model1.h5")
    print("Loaded model from disk")
    return loaded_model_co

# Load three scalars

def load_scalars():
    scaler_high = joblib.load("scaler_high.save")
    scaler_medium = joblib.load("scaler_medium.save")
    scaler_low = joblib.load("scaler_low.save")

    return scaler_high,scaler_medium,scaler_low

def predictor(data_set,scaler_co,loaded_model_co):
    # Get the model and do predictions for the data given
    forecast = loaded_model_co.predict(data_set)  # Get the last 5 miniutes data for forecast
    # print (forecast)
    # inverse_forecast = forecast.reshape(forecast.shape[0] * num_forecasts, 10) # 10 denotes the number of components
    # print (forecast.shape)
    inverse_forecast = forecast.reshape(10, 13)
    inverse_forecast = scaler_co.inverse_transform(inverse_forecast)


    # print (inverse_forecast)

    inverse_forecast_features = inverse_forecast.reshape(forecast.shape[0], 130)

    energy_value_actual_co = 0
    for j in range(0, inverse_forecast_features.shape[1]):
        #if j in [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 6, 16, 26, 36, 46, 56,
        #         66, 76, 86, 96, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97]:
        #    continue
        #else:
        energy_value_actual_co = energy_value_actual_co + inverse_forecast_features[0, j]

    # Return the energy values exclude that of the componenets
    return energy_value_actual_co



if __name__ == '__main__':
    df_high,df_medium,df_low  = equalize_indexes()
    f = open("adaptation.csv", "w").close()

    count = 0
    # Perform the adaptation for 1 day
    last_read = 0

    current_df_co = df_low.iloc[last_read:last_read + 5]

    current_df_values_co = current_df_co.values

    test_Set_co = current_df_values_co.reshape((1, 5, 13))

    loaded_model = load_model()
    scalar_high,scalar_medium,scalar_low = load_scalars()

    scalar = scalar_low
    medium =  False

    while (count<(288)):
        # Perform for 1 day = 1440 minutes with batches of 5
        energy_value_co = predictor(test_Set_co,scalar,loaded_model)
        if medium is True:
            energy_value_co = energy_value_co + 7
        print (energy_value_co)

        if energy_value_co  >= 21.0:
            # High energy state
            current_df_co = df_high.iloc[last_read:last_read + 5]
            current_df_values_co = current_df_co.values
            test_Set_co = current_df_values_co.reshape((1, 5, 13))
            scalar = scalar_high
            with open('adaptation.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            print(last_read)
            print ("high")
            medium = False


        elif energy_value_co >= 18.0 and energy_value_co < 21.0:
            current_df_co = df_medium.iloc[last_read:last_read + 5]
            medium = True
            current_df_values_co = current_df_co.values
            test_Set_co = current_df_values_co.reshape((1, 5, 13))
            scalar = scalar_medium
            with open('adaptation.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            print(last_read)
            print ("average")

        elif energy_value_co < 18.0:
            print ("low")
            current_df_co = df_low.iloc[last_read:last_read + 5]
            current_df_values_co = current_df_co.values
            test_Set_co = current_df_values_co.reshape((1, 5, 13))
            scalar = scalar_low
            with open('adaptation.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            print(last_read)
            medium = False
        last_read = last_read + 5

        count = count+1


