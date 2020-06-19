_Author_ = "***********"

# LSTM Learner to predict forecast Energy Consumption for the next 10 seconds
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
#from keras.layers import Flatten
from tensorflow.python.keras.layers import LSTM

from math import sqrt
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
import tensorflow as tf
from numpy import array
from sklearn.externals import joblib
import numpy as np
from Initializer import Initialize
from keras import backend as K
import time
#from keras.layers.core import Dense, Activation, Dropout
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
#from keras.models import model_from_json
from tensorflow.python.keras.models import model_from_json

init_object = Initialize()

class LSTM_Learner():
    # Class to perform Model Creation and Learning
    # Consists of different functions for normalization and data arrangement

    # date-time parsing function for loading the dataset

    def s_mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred)))) * 100

    def mean_absolute_scaled_error(self, y_true, y_pred, naive_mod_mae):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        error = y_true - y_pred
        return np.mean(abs(error) / naive_mod_mae)


    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # Load the dataset into the memory and start the learning
    def read_data(self,filepath):
        # Takes as input the path of the csv which contains the descriptions of energy consumed

        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        aggregated_series =  aggregated_df.values # Convert the dataframe to a 2D array and pass back to the calling function


        return aggregated_series

    # convert time series into supervised learning problem
    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        # Convert the dataset into form such that the data set is shifted for different timesteps until the number of forecasts
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # Concatenate all columns into one dataframe
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        #print(agg)
        return agg


    # create a differenced series
    def difference(self,dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)


    # transform series into train and test sets for supervised learning
    def prepare_data(self,series,prev_steps, num_forecasts):
        # Normalize the data values so that they all are in the same range
        # Convert the problem into a supervised Learning problem
        # rescale values to -1, 1

        #scaler = MinMaxScaler(feature_range=(-1, 1)) # Energy value is always posetive
        scaler = RobustScaler()
        #scaler = joblib.load("./model/scaler_h15.save")
        scaled_values = scaler.fit_transform(series)

        joblib.dump(scaler, "./model/scaler_h15_robust_v2.save")


        # transform into supervised learning problem X, y
        # prev_steps indicates the number of time steps that needs to be taken into consideration for a decision
        # n_seq denotes the number of values to be predicite

        supervised = self.series_to_supervised(scaled_values, prev_steps, num_forecasts)
        supervised_values = supervised.values
        return supervised_values,scaler



    def generate_train_test_data(self,series,propotion_value,num_obs,num_features):
        # Takes the series and converts it into a training and testing set based on the percentatge of division as
        # indicated by the propotion_value parameter
        # num_features indicate the number of features to be predicted
        # num_obs indicate the total number of observations

        test_count = int(len(series) * propotion_value)  # Integer value for representing the count
        print (test_count)
        train_data = series[:test_count, :]
        test_data = series[test_count:, :]

        # split into input and outputs
       # print (train_data[:, -5:-1])

        # No of observations is 60 and number of features is 12
        print ("Observations ",test_count)
        train_X, train_y = train_data[:, :num_obs],  train_data[:, num_obs:]
        test_X, test_y = test_data[:, :num_obs], test_data[:, num_obs:]
        #test_X, test_y = test_data[:, :-num_features], test_data[:, -num_features: -1]

        print (train_X.shape)
        print (train_y.shape)
        return train_X,train_y, test_X,test_y


    def reshape_test_train_dataset(self,train_X,test_X,lag,num_features):
        # Reshape the training and testing set as required by the Keras LSTM Network
        # reshape input to be 3D [samples, timesteps, features]
        #print(train_X.shape, len(train_X), train_y.shape)
        #num_features = 12
        train_X = train_X.reshape((train_X.shape[0], lag, num_features))
        test_X = test_X.reshape((test_X.shape[0], lag, num_features))
        return train_X,test_X




    # fit an LSTM network to training data
    def create_model(self,train_X,num_features):
        # reshape training into [samples, timesteps, features]
        # Design the LSTM Network
        model = Sequential()
        model.add(LSTM(293, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
        model.add(Dropout(0.2))
        #print (train_X.shape[1])
        #print(train_X.shape[2])
        #model.add(LSTM(units=55))
        model.add(Dense(num_features))  # Output shall depend on the number of features that needs to be predicted
        #model.add(Activation('relu'))
        #model.compile(loss='mae', optimizer='adam')
        model.compile(loss='mean_squared_error', optimizer='adam')
        print (model.summary())
        #time.sleep(3)
        return model




    # make one forecast with an LSTM,
    def forecast_lstm(self,model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]


    # evaluate the persistence model
    def make_forecasts(self,model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts


    # invert differenced forecast
    def inverse_difference(self,last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        return inverted


    # inverse data transform on forecasts
    def inverse_transform(self,series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted


    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(self,test_X, forecasts, n_lag, n_seq):
        # First transform the forecasts and series
        # Convert back to the orignal dimension
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        # Compare with original values
        # inverse transform both the test set and the forecasts



    # plot the forecasts in the context of the original dataset
    def plot_forecasts(self,series, forecasts, n_test):
        # plot the entire dataset in blue
        pyplot.plot(series.values)
        # plot the forecasts in red
        for i in range(len(forecasts)):
            off_s = len(series) - n_test + i - 1
            off_e = off_s + len(forecasts[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            yaxis = [series.values[off_s]] + forecasts[i]
            pyplot.plot(xaxis, yaxis, color='red')
        # show the plot
        pyplot.show()



if __name__ == '__main__':
    learner = LSTM_Learner()
    # load dataset
    aggregated_series = learner.read_data(init_object.data_path + init_object.data_file)

    energy_sum  = 0 # Keep a track on the inital sum of the energy
    num_forecasts = 15
    prev_steps = 10
    # Example
    # 6 components 3 prev steps will have 6*3 = 18 steps + 1*6 = 24
    # 6*10 = 60 + (24)
    init_object.num_features = init_object.num_features * num_forecasts
    #if num_forecasts > 1:
    #    number_obs = prev_steps*init_object.component_count + init_object.component_count *(num_forecasts-1)
    #else:
    number_obs = prev_steps*init_object.component_count # This will give the total number of observations

    values,scaler = learner.prepare_data(aggregated_series,prev_steps=prev_steps,num_forecasts=num_forecasts)
    print (values.shape)

    # split into input and outputs


    train_X, train_y, test_X, test_y = learner.generate_train_test_data(values,num_features=init_object.num_features,propotion_value=init_object.propotion_value,num_obs=number_obs)
    # Reshaped values
    #print(train_X.shape)

    # If the number of forecasts = 1 then
    #lag = (num_forecasts-1) + prev_steps
    lag = prev_steps
    train_X,test_X = learner.reshape_test_train_dataset(train_X,test_X,lag,num_features=22)

    # Model the LSTM Network
    print (train_X.shape[1],train_X.shape[2])
    print (init_object.num_features)

  
    model = learner.create_model(train_X,init_object.num_features)

    # fit network

    """
    history = model.fit(train_X, train_y, epochs=init_object.epochs, batch_size=init_object.batch_size, validation_data=(test_X, test_y), verbose=1,
                         shuffle=False)
    # plot history
    
    model_json = model.to_json()
    with open(init_object.model_path + "model_lstm_energy_reactive_stdScaler.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(init_object.model_path + "model_lstm_energy_stdScaler.h5")
    print("Saved model to disk")
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()
    forecast = model.predict(test_X)
    """

    #start_nw = time.time()

    model_path = "./model/"
    json_file_energy = open(model_path + "model_lstm_energy2_v5_H15_colab.json", 'r')
    loaded_model_energy_json = json_file_energy.read()
    json_file_energy.close()
    loaded_model_energy = model_from_json(loaded_model_energy_json)
    # load weights into new model
    loaded_model_energy.load_weights(model_path + "model_lstm_energy2_v5_H15_colab.h5")
    print("Loaded model from disk")
    #graph = tf.compat.v1.get_default_graph()
    #with graph.as_default():
    forecast = loaded_model_energy.predict(test_X)

    #print(forecast)
    #print(test_X)

    test_X = test_X.reshape((test_X.shape[0],number_obs))


    #print (test_X.shape)
    #print (forecast.shape)

    #time.sleep(2)


    #inverse_forecast = np.concatenate((forecast, test_X[:, 60:]), axis=1)

    #inverse_forecast = scaler.inverse_transform(inverse_forecast)
    #inverse_forecast = inverse_forecast[:, 0:init_object.num_features]

    inverse_forecast = forecast.reshape(forecast.shape[0]*num_forecasts,init_object.component_count)
    inverse_forecast = scaler.inverse_transform(inverse_forecast)

    #end_nw = time.time()

    #print(end_nw - start_nw)
    # For the test set perform the inverse transformation

    test_y = test_y.reshape((len(test_y)*num_forecasts, init_object.component_count))
    #print (test_y.shape)
    inv_y = scaler.inverse_transform(test_y)
    inv_y = inv_y[:, 0:init_object.num_features]


    #inverse_forecast =  scaler.inverse_transform(forecast)
    #print (inverse_forecast)

    rmse = sqrt(mean_squared_error(inv_y[:,1], inverse_forecast[:,1]))
    print('Test RMSE: %.3f' % rmse)


    #pyplot.plot(inv_y[:,1],label = "actual")
    #pyplot.plot(inverse_forecast[:,1],label="predicted")
    #pyplot.legend()
    #pyplot.show()

    inverse_forecast_features = inverse_forecast.reshape(forecast.shape[0],init_object.num_features)
    inv_y_features = inv_y.reshape(forecast.shape[0],init_object.num_features)


    #print (inverse_forecast_features.shape)
    #print (inv_y_features.shape)
    actual_list = []
    predicted_list = []

    # Loop through all the data and show the graph of actual vs predicted
    '''
    # Predict for a smaller set
    prev_vals_actual = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,19160.0, 19160.0,19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,19160.0, 19160.0]
    prev_vals_pred = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,19160.0, 19160.0,19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,19160.0, 19160.0]

    for i in range(0,inv_y_features.shape[0]):
        energy_value_actual = 0
        energy_value_pred  = 0
        for j in range(0,len(prev_vals_pred)):
            energy_j_actual = prev_vals_actual[j] - inv_y_features[i,j]
            energy_value_actual = energy_value_actual + energy_j_actual
            prev_vals_actual[j] = energy_j_actual
            energy_j_pred = prev_vals_pred[j] - inverse_forecast_features[i,j]
            energy_value_pred = energy_value_pred + energy_j_pred
            prev_vals_pred[j] = energy_j_pred

        actual_list.append(energy_value_actual)
        predicted_list.append(energy_value_pred)

    #print (actual_list)
    '''

    # Loop through all the data and show the graph of actual vs predicted

    # Predict for a smaller set

    print ("shape ",inv_y_features.shape[0])
    for i in range(0, inv_y_features.shape[0]):
        energy_value_actual = 0
        energy_value_pred = 0
        for j in range(0, inv_y_features.shape[1]):
            if j not in [20, 42, 64, 86, 108, 130, 152, 174, 196, 218, 240, 262, 284, 306, 328]:
            #if j not in [20,42,64,86,108,130,152,174,196,218]:
            #if j not in [20,42,64,86,108] and j<110:
            #if j < 22 and j!= 20:
                energy_value_actual = energy_value_actual + inv_y_features[i, j]
            #print (inv_y_features[i,j])
                energy_value_pred = energy_value_pred + inverse_forecast_features[i, j]

        actual_list.append(energy_value_actual)

        predicted_list.append(energy_value_pred)


    print (actual_list)
    #print (forecast)
    print (len(actual_list))
    #for i in range(0,inverse_forecast_features.shape[0]):
    #    energy_value = 0
    #    for j in range(0,10):
    #        energy_value = energy_value + inverse_forecast_features[i,j]

    #    predicted_list.append(energy_value)


    rmse_total = sqrt(mean_squared_error(predicted_list,actual_list))
    naive_mae = 2.64  # from Naive Forecasts
    print("RMSE ", rmse_total)
    s_mape_total = learner.s_mean_absolute_percentage_error(actual_list, predicted_list)
    print("SMAPE ", s_mape_total)
    mase_total = learner.mean_absolute_scaled_error(actual_list, predicted_list, naive_mae)
    print("MASE ", mase_total)
    owa_score = (s_mape_total + mase_total) / 2
    print("OWA ", owa_score)
    #print (predicted_list)
    #pyplot.plot(actual_list, label="actual")
    #pyplot.plot(predicted_list, label="predicted")
    #pyplot.legend()
    #pyplot.show()
    pyplot.figure(figsize=(4, 4))
    pyplot.plot(predicted_list[:100],label="forecasts",color='#ED6A5A')
    pyplot.plot(actual_list[:100], label="actual",color='#5386E4')
    pyplot.ylabel("Energy Consumption (Joules)")
    pyplot.xlabel("Time Intervals (Aggregated over 15 minutes)")
    pyplot.legend()
    pyplot.grid(True)
    # plt.axhline(y=6.0, color='green', linestyle='--', linewidth=1.5)
    pyplot.legend(loc="upper center")
    # plt.text(x=1441, y=6.0, s="Threshold Line", fontsize=8,
    #         bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))
    pyplot.savefig("./plots/rmse_lstm_H15.png", dpi=300)
    #pyplot.savefig("rmse_plot_energy.png")
    #pyplot.axis([10, 100, 0.1, 0.3])
    #pyplot.show()
    #test_dict  = aggregated_series[:,15,:]
    #test_df = Series(test_dict.items())

    #test_vals = test_df.values
    #test_vals = test_vals.reshape(1,1,6)

    #forecast = model.predict(test_vals)
    #print (forecast)


    # Iterate through the forecast to find the sum of the forecasted energy


