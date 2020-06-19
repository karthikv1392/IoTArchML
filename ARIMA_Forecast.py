_author_ = "Karthik Vaidhyanathan"
# Class that implements the naive 2 forecast method - Naive Seasonal

from Initializer import Initialize
from forecast_x import forecast_x as fx  # Package for the benchmark forecasts
from datetime import datetime
from pandas import read_csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from Custom_Logger import logger
import numpy
from pandas import Series

init_object = Initialize()

class ARIMA_Forecasts():
    def __init__(self):
        # Initialize the variables
        self.data_path = init_object.data_path + init_object.data_file
        # date-time parsing function for loading the dataset

    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # create a differenced series
    def difference(self,dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]

    def arima_forecastor(self):
        # performs the basic naive forecast F_t = y_t - 1
        # Reads the csv file into a dataframe and perform forecasts for each series to get the total sum

        freq = 5 # Data every minute
        horizon = 10 # Look ahead steps

        aggregated_df = read_csv(self.data_path, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        # df_no_constants = aggregated_df.loc[:, (aggregated_df != aggregated_df.iloc[0]).any()]
        actual_energy_list = []
        data_list = []
        energy_sum = 0
        for i, row in aggregated_df.iterrows():
            for j, column in row.iteritems():
                if j not in ["S21"]:
                    energy_sum = energy_sum + column

            actual_energy_list.append(energy_sum) ## Directly append to main list
            # print (energy_sum)
            energy_sum = 0
            #if (len(data_list) == 10):
            #    actual_energy_list.append(data_list)
            #    data_list = []

        test_size = int(0.8 * len(actual_energy_list))  # Divide into train and test

        test_set = actual_energy_list[:-test_size]

        forecast_list = []
        history_list = []
        history_list = actual_energy_list
        history_data = Series(history_list).values
        minutes_day = 1440
        differenced = self.difference(history_data,minutes_day)
        model = ARIMA(differenced, order=(10, 0, 1))
        model_fit = model.fit(disp=1)

        history_list.append(test_set[0]) ## Add the first element
        logger.info("starting to forecast the test set")
        #for elem in test_set:
            # For each set of 10 forecasts perform the forecast for the next point and add it to list
            #history_list.extend(elem)
        #    fit1 = ARIMA(history_list,order=(5,1,0)).fit(disp=1)

            # Applying any the model from the package

        #    forecast_list_vals = fit1.forecast()
            #print(forecast_list_vals)

            #for i in forecast_list_vals:
        #    forecast_list.append(forecast_list_vals[0]) # Rolling forecast
            # forecast_sum = sum(forecast_list_vals)
            # forecast_list.append(forecast_sum)  # Sum of the expected energy to make the comparison
        #    history_list.append(elem)
        #logger.info("finished the test set forecast, last element forecast remaining")

        #fit1 = ARIMA(history_list, order=(5, 1, 0)).fit(disp=0)

        # Applying any the model from the package
        #forecast_list_vals = fit1.forecast()
        #forecast_list.append(forecast_list_vals[0])  # Rolling forecast

        print(forecast_list)
        logger.info(forecast_list)
        # Compute the actual sum list
        actual_list = []
        for elem in test_set:
            # print (elem_list)
            #for i in elem_list:
            actual_list.append(elem)

        print(actual_list)

        logger.info("*********************************************")
        logger.info("length of forecast list", len(forecast_list))
        logger.info("length of actual list", len(actual_list))


        # print (len(actual_list))
        # print (len(forecast_list))
        try:
            rmse_total = sqrt(mean_squared_error(forecast_list, actual_list))
            print(rmse_total)
            logger.info("RMSE Value ", rmse_total)
            pyplot.plot(forecast_list[:100], label="predicted", linewidth=2.0, color='#ED6A5A')
            pyplot.plot(actual_list[:100], label="actual", linewidth=2.0, color='#5386E4')
            # pyplot.legend()
            pyplot.ylabel("Energy Consumption (Joules)")
            pyplot.xlabel("Time in Minutes")
            pyplot.grid(True)
            # plt.axhline(y=1.58, color='green', linestyle='--', linewidth=1.5)
            pyplot.legend(loc="upper left")

            pyplot.savefig("./plots/rmse_plot_arima.png", dpi=300)

        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    naive_forecast_obj = ARIMA_Forecasts()
    naive_forecast_obj.arima_forecastor()