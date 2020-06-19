_Author_ = "Karthik Vaidhyanathan"

# Script to merge two dataframes and randomly sort the values

from datetime import datetime
from pandas import read_csv
from Initializer import Initialize
import pandas as pd

init_object = Initialize()


class Data_Merger():
    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        # Load the dataset into the memory and start the learning

    def read_data(self, filepath):
        # Takes as input the path of the csv which contains the descriptions of energy consumed

        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        aggregated_series = aggregated_df.values  # Convert the dataframe to a 2D array and pass back to the calling function

        return aggregated_series,aggregated_df



data_merger_obj = Data_Merger()

if __name__ == '__main__':
    aggregated_series, df_object1 = data_merger_obj.read_data(init_object.data_path + "aggregate_energy_test.csv")
    aggregated_series2, df_object2 = data_merger_obj.read_data(init_object.data_path + "aggregate_energy_master.csv")

    frames = [df_object1,df_object2]
    result = pd.concat(frames)
    print (len(result))
    result.sample(frac=1)
    #print (result)
    result.to_csv(init_object.data_path+"aggregate_energy_historical_40days.csv",index=True)

