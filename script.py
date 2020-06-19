# To manage anomalies in dataset

import pandas as pd

data_df = pd.read_csv("data/aggregate_energy_test_15csv",
                      sep=",",
                      index_col="timestamp", nrows=1440)  # Read the proccessed data frame

df_list = []
for value in range(1,23):
    df = data_df['S'+str(value)].replace(to_replace=0.0, method='ffill')
    df_list.append(df)
    print (df)

horizontal_stack = pd.concat(df_list, axis=1)
print (horizontal_stack)



#print (data_df)

horizontal_stack.to_csv("./data/aggregate_15_test.csv",index="timestamp")
