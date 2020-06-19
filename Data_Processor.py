_Author_ = "Karthik Vaidhyanathan"

# The data processing class that can be used by all other modules

from Initializer import Initialize
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

init_object = Initialize()

class Data_Processor():
    # First collect the data required
    def __init__(self):
        self.data_path = init_object.data_path
        self.data_file = init_object.data_file

    def parser(self, x):
        return datetime.strptime(x,'%Y-%m-%d %H:%M:%S')


    def read_data(self):
        # Read the csv data. Input has to be a csv data. This is read from the init

        # Takes as input the path of the csv which contains the descriptions of energy consumed
        filepath = self.data_path + self.data_file
        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        aggregated_series = aggregated_df.values  # Convert the dataframe to a 2D array and pass back to the calling function

        return aggregated_series

    def process_energy_data(self):
        # Get all the data and plot the full series with the total energy data
        aggregated_df = read_csv(self.data_path+self.data_file, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        # df_no_constants = aggregated_df.loc[:, (aggregated_df != aggregated_df.iloc[0]).any()]
        actual_energy_list = []
        data_list = []
        energy_sum = 0
        for i, row in aggregated_df.iterrows():
            energy_sum = 0
            for j, column in row.iteritems():
                # if j not in ["S21","S7","S3","S4","S8","S11","S14", "S17","S15","S18","S2","S16"]:
                if j not in ["S21"]:
                    energy_sum = energy_sum + column

            data_list.append(energy_sum)
            if(len(data_list)== 10):
                actual_energy_list.append(sum(data_list))
                data_list = []

        print (actual_energy_list)
        print (len(actual_energy_list))
        pyplot.plot(actual_energy_list[100:200], label="energy")
        pyplot.legend()
        pyplot.show()

if __name__ == '__main__':
    data_obj = Data_Processor()
    data_obj.process_energy_data()