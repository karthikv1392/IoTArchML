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
#import keras
#from keras.models import model_from_json
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

    #print (df_medium)
    #print ("SSSSSSSSSSSS")
    #print (df_low)
    return df_high,df_medium,df_low

def total_energy(data_set):

    energy_value_actual_co = 0
    for i, row in data_set.iterrows():
        for j, column in row.iteritems():
            energy_value_actual_co = energy_value_actual_co + column

    # Return the energy values exclude that of the componenets
    #print (energy_value_actual_co)
    return energy_value_actual_co



if __name__ == '__main__':
    df_high,df_medium,df_low  = equalize_indexes()

    count = 0
    # Perform the adaptation for 1 day
    last_read = 0

    current_df_co = df_low.iloc[last_read:last_read + 10]


    while (count<(288)):
        # Perform for 1 day = 1440 minutes with batches of 5
        energy_value_co = total_energy(current_df_co)
        #print (energy_value_co)
        if energy_value_co >= 21.0:
            # High energy state
            current_df_co = df_high.iloc[last_read:last_read + 10]
            #scalar = scalar_high
            with open('adaptation_reactive.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            #print(last_read)
            print ("high")


        elif energy_value_co >= 18.0 and energy_value_co < 21.0:
            current_df_co = df_medium.iloc[last_read:last_read + 10]
            #scalar = scalar_medium
            with open('adaptation_reactive.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            #print(last_read)
            print ("average")



        elif energy_value_co < 18.0:
            print ("low")
            current_df_co = df_low.iloc[last_read:last_read + 10]
            #scalar = scalar_low
            with open('adaptation_reactive.csv', 'a') as f:
                (current_df_co).to_csv(f, header=False)
            #print(last_read)
        last_read = last_read + 10

        count = count+1


