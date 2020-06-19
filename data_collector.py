_Author_ = "*********"

# This will parse the log file of CupCarbon and generate the data

from configparser import ConfigParser

import json

import csv
import pandas as pd

from datetime import datetime
from datetime import timedelta
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from Initializer import Initialize

CONFIG_FILE = "settings.conf"
CONFIG_SECTION = "settings"


init_object = Initialize()
class DataLoader():
    # class to load the csv data to json
    # This will also support loading real-time simulation data from kafka queue for prediction
    data_path = ""
    energy_val = 19160.0 # starting energy value as per CupCarbon
    component_count = 0 # The total number of sensors for which the monitoring needs to be done
    data_file = ""
    json_path = ""
    def __init__(self):
        # Initialize the configurations
        self.data_path = init_object.data_path
        self.data_file = init_object.data_file
        self.json_path = init_object.json_path
        self.energy_val = float(init_object.energy_val)
        #self.component_count = int(init_object.component_count)
        self.component_count = 22

    def load_data_history(self):
        # Loads the historical csv file to the json

        sensor_json = {}
        prev_vals = []   # A list to store the energy values of the previous state
        # For every sensor, create a key and then insert time and energy as pairs
        #df = pd.read_csv(self.data_path + self.data_file,sep=";")   # Load the csv into a dataframe
        df = pd.read_csv(init_object.energy_path + init_object.energy_file,sep=";")   # Load the csv into a dataframe
        #print (df)
        max_time = (max(df["Time (Sec)"]))  # Find the maximum seconds for which the simulation was done.
        print (max_time)
        start_time = datetime.now() - timedelta(seconds=max_time)    # The maximum time will allow us to create a mapping for timestamp
        print (start_time)
        # Convert this data frame to a new dataframe with proper timestamps
        # Read the csv and convert into a json which can be used by time series databases

        new_df_dict = {}
        new_df_dict["timestamp"] = []
        #print (df)
        #with open(self.data_path + self.data_file) as csvfile:
        sensor_key_dict = {}

        with open(init_object.energy_path + init_object.energy_file) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';',)
            count = 0
            for row in csvreader:
                sen_count = 1
                if count == 0:
                    #time.sleep(10)
                    key_start = "S"
                    key_count  = 1
                    print (row)
                    #time.sleep(10)
                    for key in row[1:]:
                        # Create a mapping to keep all the mappings uniform
                        if key!="":
                            sensor_key_dict[key] = key_start+ str(key_count)
                            key_count+=1

                    print (sensor_key_dict)
                    #time.sleep(10)
                    for keys in row[1:]:
                        if keys != "":
                            sensor_json[sensor_key_dict[keys]] = []
                            new_df_dict[sensor_key_dict[keys]] = []
                            prev_vals.append(self.energy_val)  # all will have the same amount of energy

                #print (len(prev_vals))
                if count >=1:
                    time_value = start_time + timedelta(milliseconds=float(row[0])*1000)
                    #time_value = start_time + timedelta(seconds=float(row[0]))
                    timestamp = int(time_value.timestamp() * 1000)
                    #timestamp = int(time_value.timestamp())
                    #print (timestamp)
                    #if (timestamp) in new_df_dict["timestamp"]:
                    #    print ("exist " + str(count))
                    #else:
                    new_df_dict["timestamp"].append(timestamp)

                    #print (prev_vals)
                    #time.sleep(2)
                    while sen_count <= self.component_count:
                        time_energy_pair = [] # Create a time energy pair
                        #time_value = start_time + timedelta(seconds=float(row[0]))
                        #print (time_value)
                        # Convert the timestamp to epoch timestamp
                        #timestamp = time.mktime(time_value.timetuple())
                        #timestamp = int(time_value.timestamp() *1000)    # Get timestamp in milliseconds
                        # Normalize the data value for energy

                        # Energy value should be subtracted from the previous
                        #print (prev_vals[sen_count-1])x
                        #normalized_energy_val = round(
                        #float((self.energy_val - float(row[sen_count])) / self.energy_val), 5) * 1000
                        normalized_energy_val =  prev_vals[sen_count-1] - float(row[sen_count])
                        #print(normalized_energy_val)
                        #time.sleep(2)
                        #print (prev_vals[sen_count-1])
                        #normalized_energy_val =  (prev_vals[sen_count-1] - float(row[sen_count])) * 1000
                        prev_vals[sen_count-1] =  float(row[sen_count])    # Re assign the values
                        #normalized_energy_val =  round(prev_vals[sen_count-1] - float(row[sen_count]),2)
                        #normalized_energy_val =  round(prev_vals[sen_count-1] - float(row[sen_count]),2)

                        time_energy_pair.append(timestamp)
                        time_energy_pair.append(float(row[sen_count]))
                        sensor_json["S"+str(sen_count)].append(time_energy_pair)
                        new_df_dict["S" + str(sen_count)].append(normalized_energy_val)
                        sen_count += 1
                    #print (prev_vals)
                    #time.sleep(1)
                count += 1
                #print (count)

        
        # Now the sensor_json will have all the data stored store it in the data folder
        #print (sensor_json)

        data_json_file = open(self.json_path + "data.json","w")
        json.dump(sensor_json,data_json_file)

        #print (new_df_dict)

        # Convert the new df dict to a pandas data frame

        data_frame = pd.DataFrame(new_df_dict)

        # Send back the data_frame to the calling function and save as a new csv file
        data_frame.to_csv(self.data_path + "processed_data.csv",index=False)

        #print (process_df)

        #print (process_df.timestamp[])
        #resample_index = pd.date_range(start=process_df.timestamp[0], end=max(process_df.timestamp), freq='1s')
        #dummy_frame = pd.DataFrame(process_df, index=resample_index, columns=process_df.columns)


        # Return the data frame to
        return data_frame


    def process_data(self,data_frame):
        # Takes a dataframe as argument
        ds = pd.to_datetime(data_frame["timestamp"],
                            unit='ms')  # Convert the timestamps to new value to get the aggregated time stamp

        column_list = data_frame.columns.values
        process_df = data_frame[[column_list[1]]].copy() # Ignore the timestamp column
        for index in column_list:
            # Loop through all the columns and keep adding them
            if index != "timestamp":
                process_df[index] = data_frame[[index]]

        process_df.index = ds  # The index will go as the timestamp values
        #process_df["S2"] = data_frame[["S2"]]
        #process_df["S3"] = data_frame[["S3"]]
        #process_df["S4"] = data_frame[["S4"]]
        #process_df["S5"] = data_frame[["S5"]]
        #process_df["S6"] = data_frame[["S6"]]
        #process_df.index = ds
        aggregate_df = process_df.resample('1T').sum() # Summing up the energy values for every second frequency

        aggregate_df.to_csv(self.data_path +"aggregate_energy_proactive_15_decision_10.csv",index=True)


    def plot_generator(self):
        # Generate plots for each of the files
        file_co = init_object.data_path + "/aggregate_energy_CO_5Dec.csv"
        file_su = init_object.data_path + "/aggregate_energy_SU_5Dec.csv"
        file_sc = init_object.data_path + "/aggregate_energy_SC_5Dec.csv"
        file_ada = init_object.data_path + "/aggregate_energy_ada_29Nov.csv"
        file_rlmc = init_object.data_path + "/aggregate_energy_rlmc_4Dec.csv"
        file_mc = init_object.data_path + "/aggregate_energy_mc_5Dec.csv"
        file_rl = init_object.data_path + "/aggregate_energy_rl_5Dec.csv"

        collect_org = []
        df_collect_org = pd.read_csv(file_co, sep=",",
                                     index_col="timestamp")  # Read the proccessed data frame
        df_collect_series = df_collect_org.values
        # print(df_collect_org)
        print(type(df_collect_series))
        #prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1441):
            energy_value = 0
            for j in range(0, 22):
                if j not in [1, 15,16,17,18,19,20]:
                    #energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_collect_series[i, j]

            collect_org.append(energy_value)

        print(sum(collect_org))

        su_org = []
        df_su_org = pd.read_csv(file_su, sep=",",
                                     index_col="timestamp")  # Read the proccessed data frame
        df_su_series = df_su_org.values
        # print(df_collect_org)
        print(type(df_su_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1441):
            energy_value = 0
            for j in range(0, 17):
                if j not in [4,16]:
                #   energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_su_series[i, j]

            su_org.append(energy_value)

        print(sum(su_org))

        sc_org = []
        df_sc_org = pd.read_csv(file_sc, sep=",",
                                index_col="timestamp")  # Read the proccessed data frame
        df_sc_series = df_sc_org.values
        # print(df_collect_org)
        print(type(df_sc_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1441):
            energy_value = 0
            for j in range(0, 17):
                if j not in [4,16]:
                    #energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_sc_series[i, j]

            sc_org.append(energy_value)

        print(sum(sc_org))

        ada_org = []
        df_ada_org = pd.read_csv(file_ada, sep=",",
                                index_col="timestamp")  # Read the proccessed data frame
        df_ada_series = df_ada_org.values
        # print(df_collect_org)
        print(type(df_ada_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1441):
            energy_value = 0
            for j in range(0, 22):
                if j not in [1, 15,16,17,18,19,20]:
                    #   energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_ada_series[i, j]

            ada_org.append(energy_value)

        print(sum(ada_org))



        ##### For rlmc

        rlmc_org = []
        df_rlmc_org = pd.read_csv(file_rlmc, sep=",",
                                 index_col="timestamp")  # Read the proccessed data frame
        df_rlmc_series = df_rlmc_org.values
        # print(df_collect_org)
        print(type(df_rlmc_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1440):
            energy_value = 0
            for j in range(0, 22):
                if j not in [1, 15, 16, 17, 18, 19, 20]:
                    #   energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_rlmc_series[i, j]

            rlmc_org.append(energy_value)

        print("rl and mc: ",sum(rlmc_org))



        ###############################


        ##### For mc

        mc_org = []
        df_mc_org = pd.read_csv(file_mc, sep=",",
                                 index_col="timestamp")  # Read the proccessed data frame
        df_mc_series = df_mc_org.values
        # print(df_collect_org)
        print(type(df_mc_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1440):
            energy_value = 0
            for j in range(0, 22):
                if j not in [1, 15, 16, 17, 18, 19, 20]:
                    #   energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_mc_series[i, j]

            mc_org.append(energy_value)

        print("just MC: " ,sum(mc_org))



        ###############################

        ###############################

        ##### For RL

        rl_org = []
        df_rl_org = pd.read_csv(file_rl, sep=",",
                                index_col="timestamp")  # Read the proccessed data frame
        df_rl_series = df_rl_org.values
        # print(df_collect_org)
        print(type(df_rl_series))
        # prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
        #             19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]
        # prev_vals = [19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0,19160.0]

        # Total Sum
        for i in range(0, 1440):
            energy_value = 0
            for j in range(0, 22):
                if j not in [1, 15, 16, 17, 18, 19, 20]:
                    #   energy_j = prev_vals[j] - df_collect_series[i, j]
                    energy_value = energy_value + df_rl_series[i, j]

            rl_org.append(energy_value)


        print("just rl: ",sum(rl_org))

        ###############################


        collect_agg_list = []
        collect_agg_sum = 0
        count = 0
        for data in collect_org:
            collect_agg_sum = collect_agg_sum + data
            if count%10==0 and count!=0:
                collect_agg_list.append(collect_agg_sum)
                collect_agg_sum = 0

            count+=1
        print (len(collect_org))
        print(collect_agg_list)

        su_agg_list = []
        collect_agg_sum = 0
        count = 0
        for data in su_org:
            collect_agg_sum = collect_agg_sum + data
            if count % 10 == 0 and count != 0:
                su_agg_list.append(collect_agg_sum)
                collect_agg_sum = 0

            count += 1
        print(len(collect_org))
        print(su_agg_list)
        ############################################################
        sc_agg_list = []
        collect_agg_sum = 0
        count = 0
        for data in sc_org:
            collect_agg_sum = collect_agg_sum + data
            if count % 10 == 0 and count != 0:
                sc_agg_list.append(collect_agg_sum)
                collect_agg_sum = 0

            count += 1
        print(len(collect_org))
        print(sc_agg_list)
        ################################################
        ada_agg_list = []
        ada_agg_sum = 0
        count = 0
        for data in ada_org:
            ada_agg_sum = ada_agg_sum + data
            if count % 10 == 0 and count != 0:
                ada_agg_list.append(ada_agg_sum)
                ada_agg_sum = 0

            count += 1
        print(len(ada_org))
        print(ada_agg_list)
        print (sum(ada_agg_list))
        ##############################################

        rlmc_agg_list = []
        rlmc_agg_sum = 0
        count = 0
        for data in rlmc_org:
            rlmc_agg_sum = rlmc_agg_sum + data
            if count % 10 == 0 and count != 0:
                rlmc_agg_list.append(rlmc_agg_sum)
                rlmc_agg_sum = 0

            count += 1
        print(len(rlmc_org))
        print(rlmc_agg_list)
        print (sum(rlmc_agg_list))

        ##############################################

        mc_agg_list = []
        mc_agg_sum = 0
        count = 0
        for data in mc_org:
            mc_agg_sum = mc_agg_sum + data
            if count % 10 == 0 and count != 0:
                mc_agg_list.append(mc_agg_sum)
                mc_agg_sum = 0

            count += 1
        print(len(mc_org))
        print(mc_agg_list)
        print (sum(mc_agg_list))

        ##############################################

        rl_agg_list = []
        rl_agg_sum = 0
        count = 0
        for data in rl_org:
            rl_agg_sum = rl_agg_sum + data
            if count % 10 == 0 and count != 0:
                rl_agg_list.append(rl_agg_sum)
                rl_agg_sum = 0

            count += 1
        print(len(rl_org))
        print(rl_agg_list)
        print(sum(rl_agg_list))

        ##################################################

        plt.plot(collect_agg_list, label="CO")
        plt.plot(sc_agg_list, label="SU")
        plt.plot(su_agg_list, label="SC")
        #plt.plot(ada_agg_list, label="Random")
        plt.plot(rlmc_agg_list, label="RLMC")
        plt.plot(mc_agg_list, label="MC")
        plt.plot(rl_agg_list, label="RL")
        # df_collect_org.plot(subplots=True)
        plt.ylabel("Energy Consumed (Joules)")
        plt.xlabel("Time in Minutes (aggregated over 10 minutes)")
        plt.grid(True)
        plt.axhline(y=10, color='green', linestyle='--', linewidth=1.5)
        plt.legend(loc="upper left")
        plt.text(x=24, y=10.0, s="Energy Goal", fontsize=7,
                 bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))

        plt.savefig("./plots/total_energy_24hours_aggregate_dec6.png",dpi=300)


        ''''
        plt.plot(collect_org, label="CO")
        plt.plot(sc_org, label="SU")
        plt.plot(su_org, label="SC")
        # plt.plot(ada_agg_list, label="Random")
        plt.plot(rlmc_org, label="RLMC",linewidth =3)
        plt.plot(mc_org, label="MC")
        plt.plot(rl_org, label="RL")
        # df_collect_org.plot(subplots=True)
        plt.ylabel("Energy Consumed (Joules)")
        plt.xlabel("Time in Minutes")
        plt.grid(True)
        plt.axhline(y=1.25, color='green', linestyle='--', linewidth=1.5)
        plt.legend(loc="upper center")
        plt.text(x=241, y=1.25, s="Energy Goal", fontsize=8,
                 bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))
        plt.savefig("./plots/total_energy_sensors_4hours_minutes_dec2.png", dpi=300)
        '''

if __name__ == '__main__':
    data_loader = DataLoader()
    data_frame = data_loader.load_data_history()
    data_loader.process_data(data_frame)
    #data_loader.plot_generator()


