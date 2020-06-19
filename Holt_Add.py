_author_ = "Karthik Vaidhyanathan"
# Class that implements the Holt method for forecasts

from Initializer import Initialize
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from pandas import read_csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import pandas as pd
import numpy as np
import json

init_object = Initialize()

class Holt_Forecasts():
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

    def holt(self):
        # performs the basic naive forecast F_t = y_t - 1
        # Reads the csv file into a dataframe and perform forecasts for each series to get the total sum

        freq = 1  # Data every minute
        horizon = 30  # Look ahead steps
        lag = 100

        aggregated_df = read_csv(self.data_path, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        # df_no_constants = aggregated_df.loc[:, (aggregated_df != aggregated_df.iloc[0]).any()]
        actual_energy_list = []
        data_list = []
        main_energy_list = []
        for i, row in aggregated_df.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                if j not in ["S21"]:
                    energy_sum = energy_sum + column

            data_list.append(energy_sum)
            # main_energy_list.append(energy_sum)
            # print (energy_sum)
            # energy_sum =0
            # if (len(data_list) == 10):
            #    actual_energy_list.append(data_list)
            #    data_list = []

        actual_energy_list = data_list
        with open(init_object.data_path + "list.json", "r") as json_file:
            test_set = json.load(json_file)

        test_set = test_set["data"]
        value = test_set[0]

        forecast_list = []
        actual_set = test_set
        # print(len(test_set))
        for index in range(0, lag, 1):
            test_set.insert(index, actual_energy_list[(len(actual_energy_list) - 1) - index])

        search_index = test_set.index(value)

        last_index = search_index  # Get the index of the element to start the operation wit

        for elem_list in test_set[last_index:]:
            # For each set of 10 forecasts perform the forecast for the next point and add it to list
            # Add damped = True
            history_list = test_set[last_index - lag:last_index]
            series = pd.Series(history_list)
            fit1 = Holt(series, exponential=False).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=True)

            # Applying any the model from the package

            forecast_list_vals = fit1.forecast(horizon)
            forecast_list_vals = forecast_list_vals.tolist()
            forecast_list.append(sum(forecast_list_vals))
            # forecast_sum = sum(forecast_list_vals)
            # forecast_list.append(forecast_sum)  # Sum of the expected energy to make the comparison
            # history_list.extend(elem_list)
            last_index = last_index + 1

        actual_list = []
        for index in range(0, len(test_set[lag:])):
            # print (elem_list)
            actual_list.append(sum(test_set[index:index + horizon]))

        print(len(actual_list))

        # print (len(actual_list))
        # print (len(forecast_list))

        rmse_total = sqrt(mean_squared_error(actual_list, forecast_list))
        naive_mae = 5.15  # from Naive Forecasts
        print("RMSE ", rmse_total)
        s_mape_total = self.s_mean_absolute_percentage_error(actual_list, forecast_list)
        print("SMAPE ", s_mape_total)
        mase_total = self.mean_absolute_scaled_error(actual_list, forecast_list, naive_mae)
        print("MASE ", mase_total)
        owa_score = (s_mape_total + mase_total) / 2
        print("OWA ", owa_score)

        pyplot.plot(forecast_list[:100], label="predicted", linewidth=2.0, color='#ED6A5A')
        pyplot.plot(actual_list[:100], label="actual", linewidth=2.0, color='#5386E4')
        # pyplot.legend()
        pyplot.ylabel("Energy Consumption (Joules)")
        pyplot.xlabel("Time in Minutes")
        pyplot.grid(True)
        # plt.axhline(y=1.58, color='green', linestyle='--', linewidth=1.5)
        pyplot.legend(loc="upper left")

        pyplot.savefig("./plots/rmse_plot_holt_add_H30.png", dpi=300)


if __name__ == '__main__':
    ses_forecast_obj = Holt_Forecasts()
    ses_forecast_obj.holt()