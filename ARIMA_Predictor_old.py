_Author_ = "Karthik Vaidhyanathan"
# Script to implement the VARIMA Model for cheking accuracy of prediction
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX,VARMAXResults
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from Initializer import Initialize
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from random import random
import pickle
import time

from Initializer import Initialize

init_object  = Initialize()

class VARIMA_Learner():

    # date-time parsing function for loading the dataset
    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    def prepare_data(self,dataframe):
        # Takes the input data frame and then outputs the sorted list
        data = list()
        for i, row in dataframe.iterrows():
            data_list = []
            for j, column in row.iteritems():
                data_list.append(column)
            data.append(data_list)
        return data


    def read_data(self,filepath):
        # Takes as input the path of the csv which contains the descriptions of energy consumed
        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)
        return aggregated_df.head(13827)

    def create_model(self,data):
        # Data is basically set of lists
        model = ARIMA(data, order=(1, 1))
        model_fit = model.fit(disp=False)
        model_fit.save("varmax_model")
        return model_fit


    def make_predictions(self):
        model = VARMAXResults.load("varmax_model")
        last_read = 13827
        y_hat = model.forecast(steps=10)[0]
        print (y_hat)
        print (y_hat.shape)

    def predict_VARIMA(self, filepath):
        # Read the csv, for every 10 rows predict the next 10 rows
        aggregated_series = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                     squeeze=True, date_parser=self.parser)

        df_no_constants = aggregated_series.loc[:, (aggregated_series != aggregated_series.iloc[0]).any()]
        total_energy_list = []

        for i, row in df_no_constants.iterrows():
            energy_sum = 0
            data_list   = []
            for j, column in row.iteritems():
                energy_sum = energy_sum + column
                data_list.append(column)
            total_energy_list.append(data_list)

        test_size = int(0.8 * len(total_energy_list))  # Divide into train and test

        # Create train and test sets
        print(total_energy_list)

        train, test = total_energy_list[0:100], total_energy_list[test_size:]



        # First add everything to the energy list
        predicted_list = []
        actual_list = []
        model_fit = None
        model = VARMAX(train, order=(2,1,0), maxiter=0)
        model_fit = model.fit(disp=True, trend="nc")
        prediction = model_fit.forecast(steps=10)[0]
        print (prediction)
        for index in range(len(test) - 11):
            prediction = model_fit.forecast(steps=10)[0]
            predicted_list.append(sum(prediction))

            obs = test[index]
            train.append(obs)  # Keep adding the new observation
            actual_sum = 0
            for j in range(index, index + 10):
                actual_sum = actual_sum + test[j]
            actual_list.append(actual_sum)
            # for data in prediction:

        model_fit.save("varima_model")
        print(actual_list)
        print(predicted_list)
        rmse_total = sqrt(mean_squared_error(predicted_list, actual_list))
        print("rmse_varima", rmse_total)

        pyplot.rc('font', family='serif', serif='Times')
        pyplot.rc('text', usetex=True)
        pyplot.rc('xtick', labelsize=8)
        pyplot.rc('ytick', labelsize=8)
        pyplot.rc('axes', labelsize=8)

        width = 5.0
        height = width / 1.618

        fig, ax = pyplot.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        pyplot.plot(predicted_list[:300], label="predicted")
        pyplot.plot(actual_list[:300], label="actual")
        pyplot.legend()
        pyplot.axis([0, 300, 6.0, 10.5])
        ax.set_ylabel('Energy Consumption (Joules)')
        ax.set_xlabel('Time (Minutes)')
        fig.set_size_inches(width, height)
        pyplot.savefig("plot_varima_energy_medFreq.pdf")

    def predict(self,filepath):
        # Read the csv, for every 10 rows predict the next 10 rows
        aggregated_series = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        total_energy_list = []

        for i, row in aggregated_series.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                energy_sum = energy_sum +column
            total_energy_list.append(energy_sum)
        test_size = int(0.8*len(total_energy_list)) # Divide into train and test

        # Create train and test sets
        print (total_energy_list)

        train, test = total_energy_list[0:test_size], total_energy_list[test_size:]
        # First add everything to the energy list
        predicted_list =[]
        actual_list = []
        model_fit = None
        for index in range(len(test)-11):
            model = ARIMA(train,order=(10,0,0))
            model_fit = model.fit(disp=False,trend="nc",maxiter=1000)
            prediction = model_fit.forecast(steps=10)[0]
            predicted_list.append(sum(prediction))

            obs = test[index]
            train.append(obs) # Keep adding the new observation
            actual_sum = 0
            for j in range(index,index+10):
                actual_sum = actual_sum + test[j]
            actual_list.append(actual_sum)
            #for data in prediction:

        model_fit.save("arima_model")
        print (actual_list)
        print (predicted_list)
        rmse_total = sqrt(mean_squared_error(predicted_list, actual_list))
        print("rmse_arima", rmse_total)

        pyplot.rc('font', family='serif', serif='Times')
        pyplot.rc('text', usetex=True)
        pyplot.rc('xtick', labelsize=8)
        pyplot.rc('ytick', labelsize=8)
        pyplot.rc('axes', labelsize=8)

        width = 5.0
        height = width / 1.618

        fig, ax = pyplot.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        pyplot.plot(predicted_list[:300], label="predicted")
        pyplot.plot(actual_list[:300], label="actual")
        pyplot.legend()
        #pyplot.axis([0, 300,6.0,10.5])
        ax.set_ylabel('Energy Consumption (Joules)')
        ax.set_xlabel('Time (Minutes)')
        fig.set_size_inches(width, height)
        pyplot.savefig("plot_arima_energy_highFreq.pdf")



    def total_energy_predictor_arima(self,dataframe):
        total_energy_list = []
        for i, row in dataframe.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                energy_sum = energy_sum +column

            total_energy_list.append(energy_sum)

        model = ARIMA(total_energy_list, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
        model_fit.save("arima_model")
        y_hat = model_fit.forecast(steps=10)[0]
        print (y_hat)


    def make_arima_predictions(self,dataframe):
        model = ARIMAResults.load("arima_model")
        total_energy_list = []
        for i, row in dataframe.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                energy_sum = energy_sum + column

            total_energy_list.append(energy_sum)
        y_hat = model.predict(exog=total_energy_list[:12])[0]
        print (y_hat)

if __name__ == '__main__':
    init_object = Initialize()

    learner = VARIMA_Learner()

    #aggregated_series = learner.read_data(init_object.data_path + "aggregated_data_normal.csv")
    #df_no_constants = aggregated_series.loc[:, (aggregated_series != aggregated_series.iloc[0]).any()]
    #data = learner.prepare_data(df_no_constants)
    #print (data)

    '''
    model_fit= learner.create_model(data)
    yhat = model_fit.forecast()
    print(yhat)
    '''

    #learner.make_predictions()
    #learner.total_energy_predictor_arima(aggregated_series)
    #learner.predict(init_object.data_path + "aggregated_data_4.sv")
    learner.predict(init_object.data_path + init_object.data_file)
    #learner.make_arima_predictions(aggregated_series)
    #learner.make_predictions()
    #learner.predict_VARIMA(init_object.data_path  +"aggregated_data_4.csv")



