_Author_ = "Karthik Vaidhyanathan"
from Custom_Logger import logger
import numpy as np
from numpy import array
from Initializer import Initialize
import random
import pandas as pd
# The purpose of this class is to implement the adaptation logic depending on the type of adaptation that needs to be executed


class Adaptation_Planner():
    def __init__(self):
        self.dict_sensor_freq_keys = {"v1EnNorm": 20000, "v1EnCrit": 5000, "v1ExNorm": 20000, "v1ExCrit": 5000,
                                "v2EnNorm": 20000, "v2EnCrit": 5000, "v2ExNorm": 20000, "v2ExCrit": 5000,
                                "v3EnNorm": 20000,
                                "v3EnCrit": 5000, "v3ExNorm": 20000, "v3ExCrit": 5000, "p1EnNorm": 60000,
                                "p1EnCrit": 10000, "p1ExNorm": 30000, "p1ExCrit": 10000, "p2EnNorm": 60000,
                                "p2EnCrit": 10000, "p2ExNorm": 30000, "p2ExCrit": 10000}
        self.sensor_id_key_map = {"S34" : "p1En","S33":"p1Ex","S42" : "p2En","S41":"p2Ex","S1":"v1En","S2":"v1Ex","S18":"v2En","S20":"v2Ex","S24":"v3En","S25":"v3Ex"}

        self.sensor_mapping = {'S1': 'S1', 'S11': 'S2', 'S17': 'S3', 'S18': 'S4', 'S2': 'S5', 'S20': 'S6', 'S24': 'S7', 'S25': 'S8', 'S26': 'S9', 'S33': 'S10', 'S34': 'S11', 'S35': 'S12', 'S41': 'S13', 'S42': 'S14', 'S43': 'S15', 'S46': 'S16', 'S47': 'S17', 'S48': 'S18', 'S49': 'S19', 'S50': 'S20', 'S51': 'S21', 'S7': 'S22'}
        self.reverse_sensor_map = {'S1': 'S1', 'S2': 'S11', 'S3': 'S17', 'S4': 'S18', 'S5': 'S2', 'S6': 'S20', 'S7': 'S24', 'S8': 'S25', 'S9': 'S26', 'S10': 'S33', 'S11': 'S34', 'S12': 'S35', 'S13': 'S41', 'S14': 'S42', 'S15': 'S43', 'S16': 'S46', 'S17': 'S47', 'S18': 'S48', 'S19': 'S49', 'S20': 'S50', 'S21': 'S51', 'S22': 'S7'}
        self.init_obj = Initialize()
        self.sensor_id_list = [] # Integere values just containing the id of the sensors
        for key in self.sensor_id_key_map:
            sensor_id = int(self.sensor_mapping[key].split("S")[1])
            self.sensor_id_list.append(sensor_id)

        # Define the energy thresholds 1.45 and 1.35§§§§§§§§§§§§§§§§§§§§§
        #self.high_power = 14.5
        #self.high_power = 22.05
        self.high_power = self.init_obj.energy_hp

        #self.base_power = 13.5
        self.base_power = self.init_obj.energy_bp

        # Define the reduction frequency values
        self.reduction_freq_normal_hp = 20000
        self.reduction_freq_critical_hp = 10000
        self.reduction_freq_normal_bp = 10000
        self.reduction_freq_critical_bp = 5000
        self.adapation_count  = 0   # Keep a count on the total adaptations performed
        self.time_count  = 0 # Keep a check on the time lapsed
        self.bp_time = 0 # If a sensor has stayed in bp for 20 instances reset this value and restore to old frequency
        self.bp_count = self.init_obj.bp_count

    def reactive(self,in_energy_list):
        # Get the list of energy consumption and decide on the adaptation
        # Change the frequency of sensors in CupCarbon

        # Ignore the index which has not to be accounted for computing the increase

        # Energy list consists of 1 lists with energy of each component
        '''
        energy_list = []
        for index in range(22):
            sum_val = 0
            for i in range (len(in_energy_list)):
                sum_val = sum_val + in_energy_list[i][index]
            energy_list.append(sum_val)

        '''
        max_value  = 0
        max_index = 0
        energy_list = in_energy_list[0]
        #energy_list = in_energy_list
        for index in range(0,len(energy_list)):
            if (index!=20):
                if energy_list[index] > max_value:
                    max_value = energy_list[index]
                    max_index = index
        # Calculate the frequency reduction and write to the text file
        frequency_map = self.dict_sensor_freq_keys.copy()
        # Calculate the data transfer frequency reduction
        total_energy_consumed = sum(energy_list)
        print ("plan")
        logger.info("Inside Adaptation Planner")
        print (total_energy_consumed)
        self.time_count += self.init_obj.lag
        if total_energy_consumed>= self.high_power:
            for index in self.sensor_id_list:
                reduction_freq_critical = 0
                reduction_freq_normal = 0
                self.adapation_count += 1

                if energy_list[index] == max_value:
                    print ("here")
                    reduction_freq_normal = self.reduction_freq_normal_hp
                    reduction_freq_critical = self.reduction_freq_critical_hp
                else:
                    reduction_percent = ((max_value - energy_list[index]) / max_value)
                    reduction_freq_normal = int(self.reduction_freq_normal_hp * reduction_percent)
                    reduction_freq_critical = int(self.reduction_freq_critical_hp * reduction_percent)

                sensor_key = "S" + str(index)  # Form the sensor id to be used to get data from the reverse map
                sensor_freq_key = self.sensor_id_key_map[self.reverse_sensor_map[sensor_key]]
                frequency_map[sensor_freq_key + "Norm"] = frequency_map[
                                                              sensor_freq_key + "Norm"] + reduction_freq_normal
                frequency_map[sensor_freq_key + "Crit"] = frequency_map[
                                                              sensor_freq_key + "Crit"] + reduction_freq_critical

            # Write the adaptation to the file
            write_string = ""
            for key in frequency_map:
                write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
            write_string = write_string[:-1]

            text_file = open("config.txt", "w")
            text_file.write(write_string)
            text_file.close()


        elif total_energy_consumed < self.high_power and total_energy_consumed >= self.base_power:
            self.adapation_count += 1


            for index in self.sensor_id_list:
                reduction_freq_critical = 0
                reduction_freq_normal = 0
                if energy_list[index] == max_value:
                    reduction_freq_normal = self.reduction_freq_normal_bp
                    reduction_freq_critical = self.reduction_freq_critical_bp
                else:
                    reduction_percent = ((max_value - energy_list[index]) / max_value)
                    reduction_freq_normal = int(self.reduction_freq_normal_bp * reduction_percent)
                    reduction_freq_critical = int(self.reduction_freq_critical_bp * reduction_percent)

                sensor_key = "S" + str(index)  # Form the sensor id to be used to get data from the reverse map
                sensor_freq_key = self.sensor_id_key_map[self.reverse_sensor_map[sensor_key]]
                frequency_map[sensor_freq_key + "Norm"] = frequency_map[
                                                              sensor_freq_key + "Norm"] + reduction_freq_normal
                frequency_map[sensor_freq_key + "Crit"] = frequency_map[
                                                              sensor_freq_key + "Crit"] + reduction_freq_critical

            # Write the adaptation to the file
            write_string = ""
            for key in frequency_map:
                write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
            write_string = write_string[:-1]
            text_file = open("config.txt", "w")
            text_file.write(write_string)
            text_file.close()


        elif total_energy_consumed < self.base_power:
            # Means no need to perform any adaptation the original frequency remains as it is
            self.bp_time +=1
            if (self.bp_time>=self.init_obj.bp_count): # Change this depending on the lag value
                # Restore back to original frequencies
                # Write the adaptation to the file
                write_string = ""
                for key in frequency_map:
                    write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
                write_string = write_string[:-1]
                text_file = open("config.txt", "w")
                text_file.write(write_string)
                text_file.close()

        logger.info("Adaptation reactive executor")

    def proactive(self,inverse_forecast_features,energy_forecast_total,horizon=10):
        self.time_count += self.init_obj.lag
        # First form the list with the accumulated sum of each forecast for the time horizon
        #print (inverse_forecast_features)
        in_energy_list = []
        energy_value_list = []

        for j in range(0, inverse_forecast_features.shape[1]):
            energy_value_list.append(inverse_forecast_features[0, j])
            if (len(energy_value_list)==22):
                in_energy_list.append(energy_value_list)
                energy_value_list = []

        energy_list = []
        print (in_energy_list)
        for index in range(22):
            sum_val = 0
            for i in range(horizon):
                sum_val = sum_val + in_energy_list[i][index]
            energy_list.append(sum_val)


        #print (energy_list)
        print (energy_forecast_total)
        max_value  = 0
        max_index = 0
        for index in range(0,len(energy_list)):
            if (index!=20):
                if energy_list[index] > max_value:
                    max_value = energy_list[index]
                    max_index = index
        # Calculate the frequency reduction and write to the text file
        frequency_map = self.dict_sensor_freq_keys.copy()
        # Calculate the data transfer frequency reduction
        total_energy_consumed = sum(energy_list)

        print ("proactive plan")
        logger.info("Inside Adaptation Planner --proactive")
        #total_energy_consumed = total_energy_consumed + random.randint(0,3)
        total_energy_consumed = energy_forecast_total
        if total_energy_consumed>= self.high_power:
            #self.time_count += self.init_obj.lag
            self.adapation_count += 1
            for index in self.sensor_id_list:
                reduction_freq_critical = 0
                reduction_freq_normal = 0
                if energy_list[index] == max_value:
                    #print ("here")
                    reduction_freq_normal = self.reduction_freq_normal_hp
                    reduction_freq_critical = self.reduction_freq_critical_hp
                else:
                    reduction_percent = ((max_value - energy_list[index]) / max_value)
                    reduction_freq_normal = int(self.reduction_freq_normal_hp * reduction_percent)
                    reduction_freq_critical = int(self.reduction_freq_critical_hp * reduction_percent)

                sensor_key = "S" + str(index)  # Form the sensor id to be used to get data from the reverse map
                sensor_freq_key = self.sensor_id_key_map[self.reverse_sensor_map[sensor_key]]
                frequency_map[sensor_freq_key + "Norm"] = frequency_map[
                                                              sensor_freq_key + "Norm"] + reduction_freq_normal
                frequency_map[sensor_freq_key + "Crit"] = frequency_map[
                                                              sensor_freq_key + "Crit"] + reduction_freq_critical

            # Write the adaptation to the file
            write_string = ""
            for key in frequency_map:
                write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
            write_string = write_string[:-1]

            text_file = open("config.txt", "w")
            text_file.write(write_string)
            text_file.close()


        elif total_energy_consumed < self.high_power and total_energy_consumed >= self.base_power:
            self.adapation_count += 1
            for index in self.sensor_id_list:
                reduction_freq_critical = 0
                reduction_freq_normal = 0
                if energy_list[index] == max_value:
                    reduction_freq_normal = self.reduction_freq_normal_bp
                    reduction_freq_critical = self.reduction_freq_critical_bp
                else:
                    reduction_percent = ((max_value - energy_list[index]) / max_value)
                    reduction_freq_normal = int(self.reduction_freq_normal_bp * reduction_percent)
                    reduction_freq_critical = int(self.reduction_freq_critical_bp * reduction_percent)

                sensor_key = "S" + str(index)  # Form the sensor id to be used to get data from the reverse map
                sensor_freq_key = self.sensor_id_key_map[self.reverse_sensor_map[sensor_key]]
                frequency_map[sensor_freq_key + "Norm"] = frequency_map[
                                                              sensor_freq_key + "Norm"] + reduction_freq_normal
                frequency_map[sensor_freq_key + "Crit"] = frequency_map[
                                                              sensor_freq_key + "Crit"] + reduction_freq_critical

            # Write the adaptation to the file
            write_string = ""
            for key in frequency_map:
                write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
            write_string = write_string[:-1]
            text_file = open("config.txt", "w")
            text_file.write(write_string)
            text_file.close()


        elif total_energy_consumed < self.base_power:
            # Means no need to perform any adaptation the original frequency remains as it is
            self.bp_time +=1
            if (self.bp_time>=self.bp_count):
                # Restore back to original frequencies
                # Write the adaptation to the file
                write_string = ""
                for key in frequency_map:
                    write_string = write_string + key + " " + str(frequency_map[key]) + "\n"
                write_string = write_string[:-1]
                text_file = open("config.txt", "w")
                text_file.write(write_string)
                text_file.close()

        logger.info("Adaptation reactive executor")

    def tester_model(self,data_path,lag,horizon):
        # Function to check the efficiency of a model
        from sklearn.externals import joblib

        import tensorflow as tf

        from tensorflow.python.keras.models import model_from_json

        #energy_model_file_json = "model_lstm_energy2_v2_H30_colab.json"
        energy_model_file_json = "model_lstm_energy2_v3_H30_colab.json"
        #energy_model_file_json = "model_lstm_energy2_v5_H10_colab.json"
       # energy_model_file_json = "model_lstm_energy2_v1_H5_colab.json"
        #energy_model_file_h5 = "model_lstm_energy_v2_H30_colab.h5"
        energy_model_file_h5 = "model_lstm_energy2_v3_H30_colab.h5"
        #energy_model_file_h5 = "model_lstm_energy2_v5_H10_colab.h5"
        #energy_model_file_h5 = "model_lstm_energy2_v1_H5_colab.h5"


        #scalar_energy = joblib.load("./model/scaler_h30_robust_v3.save")
        scalar_energy = joblib.load("./model/scaler_h30_robust_v3.save")
        #scalar_energy = joblib.load("./model/scaler_standard_10.save")
        #scalar_energy = joblib.load("./model/scaler_h5_standard.save")


        graph = tf.get_default_graph()

        init_obj= Initialize()
        json_file_energy = open(init_obj.model_path + energy_model_file_json, 'r')
        loaded_model_energy_json = json_file_energy.read()
        json_file_energy.close()
        loaded_model_energy = model_from_json(loaded_model_energy_json)
        # load weights into new model
        loaded_model_energy.load_weights(init_obj.model_path + energy_model_file_h5)
        print("Loaded model from disk")

        adaptation_df = pd.read_csv(data_path,
                                    sep=",",
                                    index_col="timestamp")  # Read the proccessed data frame

        adaptation_df_series = adaptation_df.values
        main_energy_list = []

        forecast_list = []
        actual_list = []
        actual_main_list = []

        for i in range(0, len(adaptation_df)):
            energy_value = 0
            #for j in range(0, 22):
            list_val = adaptation_df_series[i,:]
            main_energy_list.append(list_val)
            sum_val = 0
            for index in range(0,len(list_val)):
                if index!=20:
                    sum_val = sum_val + list_val[index]
            actual_main_list.append(sum_val)

            if len(actual_main_list)==horizon:
                actual_list.append(sum(actual_main_list))
                actual_main_list = []

            if len(main_energy_list) == 20:
                #print (main_energy_list)
                predict_array = np.array(main_energy_list)
                # print (predict_array.shape)
                predict_array = scalar_energy.fit_transform(predict_array)
                predict_array = predict_array.reshape(1, lag, 22)
                with graph.as_default():
                    energy_forecast = loaded_model_energy.predict(predict_array)
                # K.clear_session()
                inverse_forecast = energy_forecast.reshape(horizon, 22)
                inverse_forecast = scalar_energy.inverse_transform(inverse_forecast)
                inverse_forecast_features = inverse_forecast.reshape(energy_forecast.shape[0], 22 * horizon)
                energy_forecast_total = 0
                for j in range(0, inverse_forecast_features.shape[1]):
                    # for j in range(0, 22*horizon): # Number of components * horizon equals inverse_forecast_Features.shape[1]
                    if j not in [20, 42, 64, 86, 108, 130, 152, 174, 196, 218, 240, 262, 284, 306, 328, 350, 372, 394,
                                 416, 438, 460, 482, 504, 526, 548, 570, 592, 614, 636, 658]:
                        energy_forecast_total = energy_forecast_total + inverse_forecast_features[0, j]
                forecast_list.append(energy_forecast_total)
                #print (energy_forecast_total)
                #main_energy_list.pop()
                #main_energy_list.pop()
                #main_energy_list.pop()
                #main_energy_list.pop()
                #main_energy_list.pop()
                main_energy_list = []
        print (actual_list)
        print (forecast_list)

if __name__ == '__main__':
   ada_plan_obj = Adaptation_Planner()

   #ada_plan_obj.reactive([[0.2546080000392976, 0.21209760000783717, 1.0735600002917636, 0.8160048001955147, 0.34579200006555766, 0.1873040000355104, 0.7779199998694821, 0.414683200095169, 0.57155760016758, 0.4832416000790545, 1.3014400003376068, 0.23820560004605795, 0.27449920008075424, 1.3814128003577935, 0.13796640000509797, 0.2656368000098155, 0.24916320000920678, 0.07207200000266312, 0.5317488001528545, 0.4250360000805813, 2.3117408000762225, 0.07298720001563197], [0.08644800001638941, 0.030888000001141336, 0.15640640004130546, 0.1202704000279482, 0.08644800001638941, 0.028816000005463138, 0.10135839998474694, 0.04782720001094276, 0.09213120002823416, 0.020191200001136167, 0.2609360000678862, 0.04322400000819471, 0.021612000004097354, 0.06321600001683692, 0.02059200000076089, 0.012355200000456534, 0.04942080000182614, 0.008236800000304356, 0.021612000004097354, 0.021612000004097354, 0.3010960000210616, 0.007677600002352847], [0.03365200000189361, 0.0185328000006848, 0.09157280002546031, 0.06404640001346706, 0.08644800001638941, 0.04322400000819471, 0.09197759998642141, 0.08866240001952974, 0.09213120002823416, 0.020191200001136167, 0.2575040000665467, 0.021612000004097354, 0.0072040000013657846, 0.06321600001683692, 0.010296000000380445, 0.012355200000456534, 0.04942080000182614, 0.014414400000532623, 0.021612000004097354, 0.021612000004097354, 0.2732128000170633, 0.007677600002352847], [0.021612000004097354, 0.012355200000456534, 0.06205600001703715, 0.03160800000841846, 0.08644800001638941, 0.04322400000819471, 0.08213919998888741, 0.1448544000304537, 0.03602000000682892, 0.020191200001136167, 0.18621600004917127, 0.014408000002731569, 0.0072040000013657846, 0.10251840002820245, 0.006177600000228267, 0.02059200000076089, 0.0370656000013696, 0.02471040000091307, 0.021612000004097354, 0.057632000010926276, 0.24392480003007222, 0.04606560001411708], [0.057632000010926276, 0.028828800001065247, 0.1503456000391452, 0.040132800007995684, 0.08644800001638941, 0.04322400000819471, 0.10135839998474694, 0.10450880002463236, 0.061420800018822774, 0.057632000010926276, 0.2035520000536053, 0.014408000002731569, 0.0072040000013657846, 0.1580400000420923, 0.006177600000228267, 0.030888000001141336, 0.03912480000144569, 0.0185328000006848, 0.021612000004097354, 0.08644800001638941, 0.3010960000283376, 0.015355200004705694], [0.03602000000682892, 0.0370656000013696, 0.18964800005051075, 0.03160800000841846, 0.08644800001638941, 0.04322400000819471, 0.10662079998292029, 0.08818880001854268, 0.09213120002823416, 0.08644800001638941, 0.2575040000665467, 0.014408000002731569, 0.0072040000013657846, 0.13667520003582467, 0.006177600000228267, 0.026769600000989158, 0.04942080000182614, 0.014414400000532623, 0.06094720001783571, 0.03602000000682892, 0.31679360002090107, 0.007677600002352847], [0.057632000010926276, 0.03912480000144569, 0.2007744000547973, 0.04036160000759992, 0.03602000000682892, 0.04322400000819471, 0.09701119998499053, 0.1033488000248326, 0.09213120002823416, 0.08644800001638941, 0.19085600004837033, 0.014408000002731569, 0.007677600002352847, 0.0871296000223083, 0.006177600000228267, 0.016473600000608712, 0.0370656000013696, 0.0185328000006848, 0.028816000005463138, 0.021612000004097354, 0.2882080000235874, 0.021612000004097354], [0.028816000005463138, 0.035006400001293514, 0.18195360004756367, 0.08177120002073934, 0.021612000004097354, 0.04322400000819471, 0.08854559998508194, 0.08866240001952974, 0.03602000000682892, 0.08644800001638941, 0.09186560002490296, 0.028816000005463138, 0.038388000011764234, 0.10506720002740622, 0.014414400000532623, 0.02059200000076089, 0.0185328000006848, 0.014414400000532623, 0.021612000004097354, 0.06094720001783571, 0.26313440001104027, 0.007677600002352847], [0.06483600001229206, 0.04118400000152178, 0.2050368000564049, 0.1223616000279435, 0.021612000004097354, 0.04322400000819471, 0.10753599998133723, 0.12237760002972209, 0.061420800018822774, 0.08644800001638941, 0.11772640002891421, 0.021612000004097354, 0.04606560001411708, 0.0871296000223083, 0.02059200000076089, 0.016473600000608712, 0.02059200000076089, 0.02059200000076089, 0.021612000004097354, 0.028816000005463138, 0.31960320000871434, 0.038388000011764234], [0.0, 0.002059200000076089, 0.007465600003342843, 0.007694400002947077, 0.014408000002731569, 0.014408000002731569, 0.009151999998721294, 0.02391360000547138, 0.0, 0.006630399999266956, 0.02391360000547138, 0.0072040000013657846, 0.0, 0.007923200002551312, 0.002059200000076089, 0.002059200000076089, 0.004118400000152178, 0.004118400000152178, 0.007677600002352847, 0.00010000000111176632, 0.027180800003407057, 0.0]]
   #ada_plan_obj.reactive([[0.2546080000392976, 0.21209760000783717, 1.0735600002917636, 0.8160048001955147, 0.34579200006555766, 0.1873040000355104, 0.7779199998694821, 0.414683200095169, 0.57155760016758, 0.4832416000790545, 1.3014400003376068, 0.23820560004605795, 0.27449920008075424, 1.3814128003577935, 0.13796640000509797, 0.2656368000098155, 0.24916320000920678, 0.07207200000266312, 0.5317488001528545, 0.4250360000805813, 2.3117408000762225, 0.07298720001563197]])
   #array_values = [[0.03261228, 0.20147511, 0.025070677, 0.04963389, 0.042082615, 0.038161885, 0.044103388, 0.042406965, 0.023435643, 0.010192189, 0.014693959, 0.006913607, 0.017476393, 0.0063530067, 0.006490946, 0.06755379, 0.042584933, 0.03867944, 0.11224689, 0.13262613, 0.12278276], [0.022174962, 0.018426806, 0.25328833, 0.035114877, 0.07026754, 0.02013903, 0.055001777, 0.044151418, 0.03880794, 0.023059769, 0.03231711, 0.010917423, 0.012180885, 0.040119015, 0.019380322, 0.016256128, 0.08535794, 0.07144405, 0.09489026, 0.06151631, 0.18531169, 0.12260266], [0.012266283, 0.026139703, 0.250402, 0.039286617, 0.080008335, 0.040278785, 0.063833326, 0.035133976, 0.024799261, 0.017487302, 0.033652816, 0.011299462, 0.013065252, 0.027617536, 0.012359146, 0.011237471, 0.084141284, 0.07634315, 0.06507502, 0.10146916, 0.20686735, 0.08953629], [0.020363735, 0.03138794, 0.26670012, 0.02995104, 0.061385162, 0.07936871, 0.046624906, 0.041914735, 0.038866285, 0.022787478, 0.02149977, 0.009270198, 0.009342437, 0.028570814, 0.013599097, 0.011875079, 0.08995503, 0.05563597, 0.07101285, 0.17093615, 0.15571004, 0.11875942], [0.0332466, 0.026751641, 0.2511808, 0.024378203, 0.049631145, 0.081187084, 0.041808035, 0.052946538, 0.04930255, 0.029084, 0.018332325, 0.009254609, 0.00799275, 0.022676265, 0.008474256, 0.008933206, 0.084500015, 0.0484733, 0.05239985, 0.16863091, 0.12844406, 0.14846772]]
   #for list1 in array_values:
   #    print (len(list1))
   #print (array_values)
   #ada_plan_obj.proactive(array_values,6.50,horizon=5)
   ada_plan_obj.tester_model("./data/aggregate_energy_noadaptation.csv",20,30)