_Author_ = "Karthik Vaidhyanathan"
_credits_ = "https://machinelearningmastery.com/"

from pandas import datetime
from pandas import read_csv

# univariate multi-step vector-output mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from Initializer import Initialize
import json
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import numpy as np

init_object = Initialize()



# Python program to make forecasts using simple multi-layer Perceptron

class MLP_Predictor():
    # Class that implementes the multi layered perceptron
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


    # Load the dataset into the memory and start the learning
    def read_data(self, filepath):
        # Takes as input the path of the csv which contains the descriptions of energy consumed

        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)
        energy_list = []
        data_list = []
        energy_sum = 0
        for i, row in aggregated_df.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                # if j not in ["S21","S7","S3","S4","S8","S11","S14", "S17","S15","S18","S2","S16"]:
                if j not in ["S21"]:
                    energy_sum = energy_sum + column
                data_list.append(energy_sum)

        energy_list = data_list
        return energy_list

    # split a univariate sequence into samples
    def split_sequence(self,sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def create_model(self,lag,next_steps):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_dim=lag))
        model.add(Dense(next_steps))
        model.compile(optimizer='adam', loss='mse')
        return model



if __name__ == '__main__':
    mlp_pred_obj = MLP_Predictor()
    lag = 10  # No of input to be considered
    next_steps = 5  # No of output to be produced
    historical_data_list = mlp_pred_obj.read_data(init_object.data_path +  init_object.data_file)
    train_x, train_y = mlp_pred_obj.split_sequence(historical_data_list,lag,next_steps)
    model = mlp_pred_obj.create_model(lag,next_steps)
    model.fit(train_x,train_y,epochs=1,verbose=1)

    model_json = model.to_json()

    with open(init_object.model_path + "model_MLP_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(init_object.model_path + "model_MLP_test.h5")
    print("Saved model to disk")

    with open(init_object.data_path + "list.json", "r") as json_file:
        test_set = json.load(json_file)

    test_set = test_set["data"]
    value = test_set[0]

    forecast_list = []
    actual_set = test_set
    # print(len(test_set))
    for index in range(0, lag, 1):
        test_set.insert(index, historical_data_list[(len(historical_data_list) - 1) - index])

    search_index = test_set.index(value)

    last_index = search_index  # Get the index of the element to start the operation wit

    for elem_list in test_set[last_index:]:
        # For each set of 10 forecasts perform the forecast for the next point and add it to list
        # Add damped = True
        test_subset = test_set[last_index - lag:last_index]
        test_subset = array(test_subset)
        test_subset = test_subset.reshape((1, lag))
        prediction_list = model.predict(test_subset, verbose=0)
        forecast_list.append(sum(prediction_list[0]))

    actual_list = []
    # actual_list  = test_list
    for index in range(0, len(test_set[lag:])):
        # print (elem_list)
        actual_list.append(sum(test_set[index:index + next_steps]))

    print(len(actual_list))

    # print (len(actual_list))
    # print (len(forecast_list))

    rmse_total = sqrt(mean_squared_error(actual_list, forecast_list))
    naive_mae = 1.64  # from Naive Forecasts
    print("RMSE ", rmse_total)
    s_mape_total = mlp_pred_obj.s_mean_absolute_percentage_error(actual_list, forecast_list)
    print("SMAPE ", s_mape_total)
    mase_total = mlp_pred_obj.mean_absolute_scaled_error(actual_list, forecast_list, naive_mae)
    print("MASE ", mase_total)
    owa_score = (s_mape_total + mase_total) / 2
    print("OWA ", owa_score)

    pyplot.plot(forecast_list[:100], label="predicted", linewidth=2.0, color='#5386E4')
    pyplot.plot(actual_list[:100], label="actual", linewidth=2.0, color='#ED6A5A')
    # pyplot.legend()
    pyplot.ylabel("Energy Consumption (Joules)")
    pyplot.xlabel("Time in Minutes")
    pyplot.grid(True)
    # plt.axhline(y=1.58, color='green', linestyle='--', linewidth=1.5)
    #pyplot.axis([0, 100, 4, 12])
    pyplot.legend(loc="upper left")

    pyplot.savefig("./plots/rmse_plot_mlp_test.png", dpi=300)