_author_ = "Karthik Vaidhyanathan"
# Class that implements the naive 2 forecast method - Naive Seasonal

from Initializer import Initialize
from forecast_x import forecast_x as fx  # Package for the benchmark forecasts
from datetime import datetime
from pandas import read_csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import json
from sklearn.metrics import mean_absolute_error
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

    def seasonal_naive(self):
        # performs the basic naive forecast F_t = y_t - 1
        # Reads the csv file into a dataframe and perform forecasts for each series to get the total sum

        freq = 5 # Data every minute
        horizon = 30 # Look ahead steps

        aggregated_df = read_csv(self.data_path, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        #df_no_constants = aggregated_df.loc[:, (aggregated_df != aggregated_df.iloc[0]).any()]
        actual_energy_list = []
        data_list = []
        for i, row in aggregated_df.iterrows():

            energy_sum = 0
            for j, column in row.iteritems():
                if j not in ["S21"]:
                    energy_sum = energy_sum + column

            data_list.append(energy_sum)
            #print (energy_sum)


        actual_energy_list = data_list

        with open(init_object.data_path + "list.json", "r") as json_file:
            test_set = json.load(json_file)

        test_set = test_set["data"]
        print(test_set)
        forecast_list = []
        history_list = actual_energy_list
        history_list.append(test_set[0])

        for elem in test_set[1:]:
            # For each set of 10 forecasts perform the forecast for the next point and add it to list
            #print (history_list)
            f_list = []
            for index in range(0,horizon,1):
                f_list.append(history_list[len(history_list)-1-index])
            #print (sum(f_list))
            print (len(f_list))
            forecast_list.append(sum(f_list))
            history_list.append(elem)
            #last_index += 1

        f_list = []
        for index in range(1, horizon, 1):
            f_list.append(history_list[len(history_list) - index ])

        forecast_list.append(sum(f_list))
        print (forecast_list)
        # Compute the actual sum list
        actual_list = []
        for index in range(0, len(test_set)):
            # print (elem_list)
            actual_list.append(sum(test_set[index:index + horizon]))

        print(len(actual_list))

        #print (len(actual_list))
        #print (len(forecast_list))

        rmse_total = sqrt(mean_squared_error(actual_list, forecast_list))
        naive_mae = 5.15 # from Naive Forecasts
        print("RMSE ", rmse_total)
        s_mape_total = self.s_mean_absolute_percentage_error(actual_list, forecast_list)
        print("SMAPE ", s_mape_total)
        mase_total = self.mean_absolute_scaled_error(actual_list, forecast_list, naive_mae)
        print("MASE ", mase_total)
        owa_score = (s_mape_total+mase_total)/2
        print("OWA ", owa_score)
        pyplot.plot(forecast_list[:100], label="predicted", linewidth=2.0, color='#5386E4')
        pyplot.plot(actual_list[:100], label="actual", linewidth=2.0, color='#ED6A5A')
        #pyplot.legend()
        pyplot.ylabel("Energy Consumption (Joules)")
        pyplot.xlabel("Time in Minutes")
        pyplot.grid(True)
        # plt.axhline(y=1.58, color='green', linestyle='--', linewidth=1.5)
        #pyplot.axis([0, 100, 4, 12])
        pyplot.legend(loc="upper left")

        pyplot.savefig("./plots/rmse_plot_naives_H30.png", dpi=300)


if __name__ == '__main__':
    naive_forecast_obj = Naive_Forecasts()
    naive_forecast_obj.seasonal_naive()