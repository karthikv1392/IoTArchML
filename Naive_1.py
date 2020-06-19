_author_ = "Karthik Vaidhyanathan"
# Class that implements the naive forecast method

from Initializer import Initialize
from forecast_x import forecast_x as fx  # Package for the benchmark forecasts
from datetime import datetime
from pandas import read_csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
import json
import numpy as np

init_object = Initialize()

class Naive_Forecasts():
    def __init__(self):
        # Initialize the variables
        self.data_path = init_object.data_path + init_object.data_file
        # date-time parsing function for loading the dataset

    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    def s_mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))) * 100


    def mean_absolute_scaled_error(self, y_true, y_pred, naive_mod_mae):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        error = y_true - y_pred
        return np.mean(abs(error) / naive_mod_mae)

    def simple_naive(self):
        # performs the basic naive forecast F_t = y_t - 1
        # Reads the csv file into a dataframe and perform forecasts for each series to get the total sum

        freq = 1 # Data every minute
        horizon = 30 # Look ahead steps

        aggregated_df = read_csv(self.data_path, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        #df_no_constants = aggregated_df.loc[:, (aggregated_df != aggregated_df.iloc[0]).any()]
        actual_energy_list = []
        data_list = []
        energy_sum = 0
        for i, row in aggregated_df.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                #if j not in ["S21","S7","S3","S4","S8","S11","S14", "S17","S15","S18","S2","S16"]:
                if j not in ["S21"]:
                    energy_sum = energy_sum + column


            data_list.append(energy_sum)
            #if (len(data_list) == 10):
            #    actual_energy_list.append(sum(data_list))
            #    data_list = []
            #print (energy_sum)
            #energy_sum = 0
            #if (len(data_list) == 10):
            #    actual_energy_list.append(data_list)
            #    data_list = []



        actual_energy_list = data_list
        #print (actual_energy_list)
        num_obs = int(len(actual_energy_list)*0.7)
        #print (actual_energy_list[num_obs+10:len(actual_energy_list)])
        test_set = []
        test_list = []
        with open(init_object.data_path + "list.json") as json_file:
            test_data_json = json.load(json_file)
            test_list = test_data_json["data"]

        print (test_list)
        temp_list = []
        for element in test_list:
            temp_list.append(element)
            if len(temp_list)==horizon:
                test_set.append(temp_list)
                temp_list = []


        forecast_list  = []
        history_list = actual_energy_list
        history_list.append(test_list[0])  # Insert the first set
        for elem in test_list[1:]:
            # For each set of 10 forecasts perform the forecast for the next point and add it to list
            f = fx.forecast(history_list, freq, horizon)
            # Applying any the model from the package

            model = f.model_naive()

            forecast_list_vals = model[2]

            #for i in forecast_list_vals:
            forecast_list.append(sum(forecast_list_vals))
            #forecast_sum = sum(forecast_list_vals)
            #forecast_list.append(forecast_sum)  # Sum of the expected energy to make the comparison
            history_list.append(elem)

        f = fx.forecast(history_list, freq, horizon)
        # Applying any the model from the package
        model = f.model_naive()
        forecast_list_vals = model[2]

        # for i in forecast_list_vals:
        forecast_list.append(sum(forecast_list_vals))
        print (len(forecast_list))


        # Compute the actual sum list
        actual_list = []
        #actual_list  = test_list
        for index in range(0,len(test_list)):
            #print (elem_list)
            actual_list.append(sum(test_list[index:index+horizon]))

        print (len(actual_list))

        #print (len(actual_list))
        #print (len(forecast_list))

        rmse_total = sqrt(mean_squared_error(actual_list,forecast_list))
        naive_mae = mean_absolute_error(actual_list, forecast_list)
        print("naive mae", naive_mae)
        print ("RMSE ", rmse_total)
        s_mape_total = self.s_mean_absolute_percentage_error(actual_list,forecast_list)
        print ("SMAPE ",s_mape_total)
        mase_total = self.mean_absolute_scaled_error(actual_list,forecast_list,naive_mae)
        print ("MASE ", mase_total)
        owa_score = (s_mape_total + mase_total)/2
        print ("OWA ",owa_score)

        print(actual_list)
        print (forecast_list)

        pyplot.plot(forecast_list[0:100], label="predicted",linewidth=2.0,color='#5386E4')
        pyplot.plot(actual_list[0:100], label="actual",linewidth=2.0, color='#ED6A5A')
        #pyplot.legend()
        pyplot.ylabel("Energy Consumption (Joules)")
        pyplot.xlabel("Time in Minutes")
        pyplot.grid(True)
        # plt.axhline(y=1.58, color='green', linestyle='--', linewidth=1.5)
        #pyplot.axis([0, 100, 4, 12])
        pyplot.legend(loc="upper left")

        pyplot.savefig("./plots/rmse_plot_naive1_H30.png", dpi=300)


if __name__ == '__main__':
    naive_forecast_obj = Naive_Forecasts()
    naive_forecast_obj.simple_naive()