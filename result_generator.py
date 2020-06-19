_Author_ = "Karthik Vaidhyanathan"
from Initializer import Initialize
import pandas as pd
from statistics import median
from statistics import mean
from matplotlib import pyplot as plt

# Script for processing the results generated by the IoTArchML approach

init_object = Initialize()


def result_generator(data_path, type, plot_path):
    # type denotes the type of the file, no adaptation, adaptation, proactve etc
    # plot_path denotes the path where the plot needs to be generated
    adaptation_df = pd.read_csv(data_path,
                          sep=",",
                          index_col="timestamp")  # Read the proccessed data frame

    adaptation_df_series = adaptation_df.values

    # Total Sum
    cumulative_energy_list  = [] # To store the minute wise energy consumption for plotting purposes
    energy_joules = 0
    for i in range(0, len(adaptation_df)):
        energy_value = 0
        for j in range(0, 22):
            if j not in [20]:
                energy_value = energy_value + adaptation_df_series[i,j]
                energy_joules = energy_joules + adaptation_df_series[i,j]

        cumulative_energy_list.append(energy_value)

        #print (energy_value)
    print (cumulative_energy_list)
    print (sum(cumulative_energy_list))
    index_list = []
    sum_value = 0

    for index in range(0,len(cumulative_energy_list)):
        sum_value = sum_value + cumulative_energy_list[index]
        if (index)%15 == 0:
            index_list.append(sum_value)
            sum_value = 0

    # The calculation below can be used to fix the energy thresholds for different horizon of times
    print (index_list)
    print (sum(index_list))
    print ("max ", max(index_list[1:len(index_list)-1]))
    print ("min ", min(index_list[1:len(index_list)-1]))
    print ("median ", median(index_list[1:len(index_list)-1]))
    print("average ", mean(index_list[1:len(index_list) - 1]))
    return cumulative_energy_list

def tester():
    reactive_data_csv = "data/aggregate_energy_proactive_15_decision_15.csv"
    data_df = pd.read_csv(reactive_data_csv,
                          sep=",",
                          index_col="timestamp", nrows=1440)  # Read the proccessed data frame
    count = 0
    energy_list = []
    data_df_series = data_df.values
    for i in range(0, len(data_df)):
        energy_value = 0
        for j in range(0, 22):
            # if j not in [20]:
            energy_value = energy_value + data_df_series[i, j]
        energy_list.append(energy_value)

    aggregate_list = generate_cumulative_sum_list(energy_list,cumul=False,interval=15)
    for value in aggregate_list:
        if (value > 17.86):
            count+=1
    print (count)

def generate_cumulative_sum_list(qos_list,cumul=True,interval=10):
    # function to generate the cumulative sum of a given list of QoS values for given intervals
    cumulative_sum_list = []
    sum = 0
    if cumul:
        for index in range(0,len(qos_list)):
            sum = sum + qos_list[index]
            if (index!=0 and index%interval==0) or (index==len(qos_list)):
                cumulative_sum_list.append(sum)

    else:
        for index in range(0, len(qos_list)):
            sum = sum + qos_list[index]
            if index != 0 and index % interval == 0:
                cumulative_sum_list.append(sum)
                sum=0
    return cumulative_sum_list

def cumulative_plot_generator():
    noadap_data_csv = "data/aggregate_energy_noadaptation.csv"
    #noadap_data_csv = "data/aggregate_energy_master.csv"
    reactive_data_csv = "data/aggregate_energy_reactive_test1.csv"
    h5_data_csv = "data/aggregate_energy_proactive_5.csv"
    h10_data_csv = "data/aggregate_energy_proactive_10.csv"
    h15_data_csv = "data/aggregate_energy_proactive_15_decision_15.csv"
    h30_data_csv = "data/aggregate_energy_proactive_30.csv"

    data_source_lists = [noadap_data_csv,reactive_data_csv,h5_data_csv,h10_data_csv,h15_data_csv,h30_data_csv]


    for data_path in data_source_lists:
        data_df = pd.read_csv(data_path,
                                    sep=",",
                                    index_col="timestamp",nrows=1440)  # Read the proccessed data frame

        data_df_series = data_df.values
        energy_list = []  # To store the minute wise energy consumption for plotting purposes
        cumul_energy_list= []
        energy_joules = 0
        for i in range(0, len(data_df)):
            energy_value = 0
            for j in range(0, 22):
                if j not in [20]:
                    energy_value = energy_value + data_df_series[i, j]
                    energy_joules = energy_joules + data_df_series[i, j]
            energy_list.append(energy_value)
        print (sum(energy_list))
        cumul_energy_list = generate_cumulative_sum_list(energy_list,interval=1)

        if "reactive" in data_path:
            print (" reactive ", sum(energy_list))
            plt.plot(cumul_energy_list, label="Reactive", linewidth=1.5, color='#800000')
        elif "noadaptation" in data_path:
            print(" noadaptation ", sum(energy_list))
            plt.plot(cumul_energy_list, label="NoAdap", linewidth=1.5, color='#70C1B3')
        elif "proactive_5" in data_path:
            print(" proactive_5 ", sum(energy_list))
            plt.plot(cumul_energy_list, label="Proactive_5", linewidth=1.5, color='#50514F')
        elif "proactive_10" in data_path:
            print(" proactive_10 ", sum(energy_list))
            plt.plot(cumul_energy_list, label="Proactive_10", linewidth=1.5, color='#ED6A5A')
        elif "proactive_30" in data_path:
            print(" proactive_30 ", sum(energy_list))
            plt.plot(cumul_energy_list, label="Proactive_30", linewidth=1.5, color='#5386E4')
        elif "proactive_15_decision_15" in data_path:
            print(" proactive_15_decision_10", sum(energy_list))
            plt.plot(cumul_energy_list, label="Proactive_15", linewidth=1.5, color='#247BA0')

    plt.legend()
    plt.ylabel("Energy Consumption (Joules)")
    plt.xlabel("Time intervals (aggregated over 10 minutes)")
    plt.grid(True)
    '''
    plt.text(x=145, y=2114, s="NoAdap", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    plt.text(x=145, y=1930, s="Reactive", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    plt.text(x=145, y=1795, s="Proactive_5", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    plt.text(x=145, y=1735, s="Proactive_30", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    plt.text(x=145, y=1692, s="Proactive_10", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    plt.text(x=145, y=1600, s="Proactive_15", fontsize=8,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.3"))
    '''
    plt.savefig("./plots/energy_cumulative_plot.png", dpi=300)

def box_plot_generator():
    noadap_data_csv = "data/aggregate_energy_noadaptation.csv"
    # noadap_data_csv = "data/aggregate_energy_master.csv"
    reactive_data_csv = "data/aggregate_energy_reactive_test1.csv"
    h5_data_csv = "data/aggregate_energy_proactive_5.csv"
    h10_data_csv = "data/aggregate_energy_proactive_10.csv"
    h15_data_csv_10 = "data/aggregate_energy_proactive_15_decision_10.csv"
    h15_data_csv = "data/aggregate_energy_proactive_15_decision_15.csv"

    h30_data_csv = "data/aggregate_energy_proactive_30.csv"

    data_source_lists = [noadap_data_csv, reactive_data_csv, h5_data_csv, h10_data_csv, h15_data_csv_10, h15_data_csv, h30_data_csv]

    energy_list_noadap = []
    energy_list_reactive = []
    energy_list_h5 = []
    energy_list_h10 = []
    energy_list_h15_10 = []
    energy_list_h15 = []
    energy_list_h30 = []

    for data_path in data_source_lists:
        data_df = pd.read_csv(data_path,
                              sep=",",
                              index_col="timestamp", nrows=1440)  # Read the proccessed data frame

        data_df_series = data_df.values
        energy_list = []  # To store the minute wise energy consumption for plotting purposes
        cumul_energy_list = []
        energy_joules = 0
        for i in range(0, len(data_df)):
            energy_value = 0
            for j in range(0, 22):
                if j not in [20]:
                    energy_value = energy_value + data_df_series[i, j]
                    energy_joules = energy_joules + data_df_series[i, j]
            energy_list.append(energy_value)
        if "noadaptation" in data_path:
            energy_list_noadap = energy_list
            print ("mean NoAdap ",mean(energy_list))
            print ("median NoAdap ", median(energy_list))
            print ("max NoAdap ", max(energy_list))
            print ("min No Adap ", min(energy_list))
        elif "reactive" in data_path:
            energy_list_reactive = energy_list
            print("mean reactive ", mean(energy_list))
            print("median reactive ", median(energy_list))
            print("max reactive ", max(energy_list))
            print("min reactive ", min(energy_list))

        elif "proactive_5" in data_path:
            energy_list_h5 = energy_list
            print("mean h5 ", mean(energy_list))
            print("median h5 ", median(energy_list))
            print("max h5", max(energy_list))
            print("min h5 ", min(energy_list))

        elif "proactive_10" in data_path:
            energy_list_h10 = energy_list
            print("mean h10 ", mean(energy_list))
            print("median h10 ", median(energy_list))
            print("max h10 ", max(energy_list))
            print("min h10 ", min(energy_list))


        elif "proactive_15_decision_15" in data_path:
            print("mean h15 ", mean(energy_list))
            print("median h15 ", median(energy_list))
            print("max h15 ", max(energy_list))
            print("min h15 ", min(energy_list))
            energy_list_h15 = energy_list

            #print (sum(energy_list_h15))
        elif "proactive_30" in data_path:
            print("mean h30 ", mean(energy_list))
            print("median h30 ", median(energy_list))
            print("max h30 ", max(energy_list))
            print("min h30 ", min(energy_list))
            energy_list_h30 = energy_list

    bp = plt.boxplot((energy_list_noadap, energy_list_reactive,
                      energy_list_h5, energy_list_h10,
                      energy_list_h15,energy_list_h30), patch_artist=True)

    colors = ['#FFE066', '#5386E4', '#50514F', '#70C1B3', '#247BA0', '#5386E4']
    # colors = ['#FFE066', '#70C1B3', '#50614F','#247BA0']
    i = 0
    for box in bp['boxes']:
        box.set(facecolor=colors[i], linewidth=2)
        i += 1
    plt.xticks([1, 2, 3, 4, 5,6], ['noAdap', 'Reactive', 'pro_h5', 'pro_h10', 'pro_h15','pro_h30'])
    # plt.xticks([1, 2, 3, 4], ['sta_gre', 'tim_Qle', 'lin_gre','lin_Qle'])
    plt.ylabel("Energy Consumption (Joules)")
    plt.xlabel("Approaches")
    plt.savefig("./plots/box_plot_adaptation_quality.png", dpi=300)

def multi_box_plot_generator():
    # Generate box plots for each of the approaches with decision periods as subfigures

    #fig, axs = plt.subplots(2, 2)

    # First time horizon 5
    h5_data_csv_small = "data/aggregate_energy_proactive_5_decision_1.csv"
    h5_data_csv_medium = "data/aggregate_energy_proactive_5_decision_3.csv"
    h5_data_csv_large = "data/aggregate_energy_proactive_5.csv"

    h10_data_csv_small = "data/aggregate_energy_proactive_10_decision_3.csv"
    h10_data_csv_medium = "data/aggregate_energy_proactive_10_decision_5.csv"
    h10_data_csv_large = "data/aggregate_energy_proactive_10.csv"

    h15_data_csv_small = "data/aggregate_energy_proactive_15_decision_5.csv"
    h15_data_csv_medium = "data/aggregate_energy_proactive_15_decision_10.csv"
    h15_data_csv_large = "data/aggregate_energy_proactive_15_decision_15.csv"

    h30_data_csv_small = "data/aggregate_energy_proactive_30_decision_5.csv"
    h30_data_csv_medium = "data/aggregate_energy_proactive_30_decision_10.csv"
    h30_data_csv_large = "data/aggregate_energy_proactive_30.csv"


    data_path_dict = {}
    data_path_dict["5"] = [h5_data_csv_small,h5_data_csv_medium,h5_data_csv_large]
    data_path_dict["10"] = [h10_data_csv_small,h10_data_csv_medium,h10_data_csv_large]
    data_path_dict["15"] = [h15_data_csv_small,h15_data_csv_medium,h15_data_csv_large]
    data_path_dict["30"] = [h30_data_csv_small,h30_data_csv_medium,h30_data_csv_large]

    h5_small = []
    h5_medium = []
    h5_large = []

    h10_small = []
    h10_medium = []
    h10_large = []

    h15_small = []
    h15_medium = []
    h15_large = []

    h30_small = []
    h30_medium = []
    h30_large = []

    for key in data_path_dict.keys():
        for data_path in data_path_dict[key]:
            data_df = pd.read_csv(data_path,
                                  sep=",",
                                  index_col="timestamp", nrows=1440)  # Read the proccessed data frame

            data_df_series = data_df.values
            energy_list = []  # To store the minute wise energy consumption for plotting purposes
            energy_joules = 0
            for i in range(0, len(data_df)):
                energy_value = 0
                for j in range(0, 22):
                    if j not in [20]:
                        energy_value = energy_value + data_df_series[i, j]
                        energy_joules = energy_joules + data_df_series[i, j]
                energy_list.append(energy_value)

            if "proactive_5_decision_1" in data_path:
                h5_small = generate_cumulative_sum_list(energy_list,cumul=False,interval=5)
                print ("h5 small", sum(h5_small))
            elif "proactive_5_decision_3" in data_path:
                h5_medium = generate_cumulative_sum_list(energy_list,cumul=False,interval=5)
                print("h5 medium", sum(h5_medium))
            elif "proactive_5.csv" in data_path:
                h5_large = generate_cumulative_sum_list(energy_list,cumul=False,interval=5)
                print("h5 large", sum(h5_large))
                print("h5 large average", mean(h5_large))
                print("h5 large median", median(h5_large))

            elif "proactive_10_decision_3" in data_path:
                h10_small = generate_cumulative_sum_list(energy_list,cumul=False,interval=10)
                print("h10 small", sum(h10_small))
            elif "proactive_10_decision_5" in data_path:
                h10_medium = generate_cumulative_sum_list(energy_list,cumul=False,interval=10)
                print("h10 medium", sum(h10_medium))
            elif "proactive_10.csv" in data_path:
                h10_large = generate_cumulative_sum_list(energy_list,cumul=False,interval=10)
                print("h10 large", sum(h10_large))
                print ("h10 large average", mean(h10_large))
                print("h10 large median", median(h10_large))

            elif "proactive_15_decision_5" in data_path:
                h15_small = generate_cumulative_sum_list(energy_list,cumul=False,interval=15)
                print("h15 small", sum(h15_small))
            elif "proactive_15_decision_10" in data_path:
                h15_medium = generate_cumulative_sum_list(energy_list,cumul=False,interval=15)
                print("h15 medium", sum(h15_medium))
            elif "proactive_15_decision_15.csv" in data_path:
                h15_large =generate_cumulative_sum_list(energy_list,cumul=False,interval=15)
                print("h15 large", sum(h15_large))
                print("h15 large average", mean(h15_large))
                print("h15 large median", median(h15_large))


            elif "proactive_30_decision_5" in data_path:
                h30_small = generate_cumulative_sum_list(energy_list,cumul=False,interval=30)
                print("h30 small", sum(h30_small))
            elif "proactive_30_decision_10" in data_path:
                h30_medium = generate_cumulative_sum_list(energy_list,cumul=False,interval=30)
                print("h30 medium", sum(h30_medium))
            elif "proactive_30.csv" in data_path:
                h30_large = generate_cumulative_sum_list(energy_list,cumul=False,interval=30)
                print("h30 large", sum(h30_large))
                print("h30 large average", mean(h30_large))
                print("h30 large median", median(h30_large))

        if key=="5":
            df1 = pd.DataFrame({'small': h5_small, 'medium': h5_medium, 'large': h5_large})

        elif key=="10":
            df2 = pd.DataFrame({'small': h10_small, 'medium': h10_medium, 'large': h10_large})

        elif key=="15":
            df3 = pd.DataFrame({'small': h15_small, 'medium': h15_medium, 'large': h15_large})

        elif key=="30":
            df4 = pd.DataFrame({'small': h30_small, 'medium': h30_medium, 'large': h30_large})

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    df1.boxplot(ax=ax1)
    ax2 = fig.add_subplot(2, 2, 2)
    df2.boxplot(ax=ax2)
    ax3 = fig.add_subplot(2, 2, 3)
    df3.boxplot(ax=ax3)
    ax4 = fig.add_subplot(2, 2, 4)
    df4.boxplot(ax=ax4)
    ax1.title.set_text('Proactive_5 (H = 5)')
    ax2.title.set_text('Proactive_10 (H = 10)')
    ax3.title.set_text('Proactive_15 (H = 15)')
    ax4.title.set_text('Proactive_30 (H = 30)')
    plt.tight_layout()
    fig.text(0.5, 0.0, 'Decision Periods', ha='center')
    fig.text(0.0, 0.5, 'Energy Consumed (Joules)', va='center', rotation='vertical')
    plt.savefig("./plots/four_box_plots.png", dpi=300)

    plt.show()

def box_plot_test():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 8, 2], 'c': [9, 1, 7]})
    df2 = pd.DataFrame({'a': [15, 23, 32], 'b': [6, 80, 2], 'c': [9, 10, 7]})
    df3 = pd.DataFrame({'a': [0.2, 0.5, 0.5], 'b': [18, 5, 2], 'c': [9, 7, 7]})
    df4 = pd.DataFrame({'a': [51, 32, 20], 'b': [4, 3, 20], 'c': [7, 2, 1]})

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    df1.boxplot(ax=ax1)
    ax2 = fig.add_subplot(2, 2, 2)
    df2.boxplot(ax=ax2)
    ax3 = fig.add_subplot(2, 2, 3)
    df3.boxplot(ax=ax3)
    ax4 = fig.add_subplot(2, 2, 4)
    df4.boxplot(ax=ax4)
    plt.show()

def log_analyzer():
    import numpy as np
    # Script to analyze the logs and to generate the adaptation count
    h5_small_log_path = "logs/IoTArchML_H5_Proactive_decisionShort.log"
    h5_medium_log_path = "logs/IoTArchML_H5_Proactive_decisionMedium.log"
    h5_large_log_path = "logs/IoTArchML_H5_Proactive.log"

    h10_small_log_path = "logs/IoTArchML_H10_Proactive_decisionShort.log"
    h10_medium_log_path = "logs/IoTArchML_H10_decisionMedium.log"
    h10_large_log_path = "logs/IoTArchML_H10_Proactive.log"

    h15_small_log_path = "logs/IoTArchML_H15_Proactive_decisionShort.log"
    h15_medium_log_path = "logs/IoTArchML_H15_Proactive_decisionMedium.log"
    h15_large_log_path = "logs/IoTArchML_H15_Proactice_decisionLarge.log"

    h30_small_log_path = "logs/IoTArchML_H30_Proactive_decisionShort.log"
    h30_medium_log_path = "logs/IoTArchML_H30_Proactive_decisionMedium.log"
    h30_large_log_path = "logs/IoTArchML_H30_Proactive.log"

    log_path_list = [h5_small_log_path,h5_medium_log_path,h5_large_log_path,h10_small_log_path,h10_medium_log_path,h10_large_log_path,
                     h15_small_log_path,h15_medium_log_path,h15_large_log_path,h30_small_log_path,h30_medium_log_path,h30_large_log_path]
    bar_y_list = []
    bar_x_list = []
    for file_path in log_path_list:
        prev_count = 0
        adaptation_count  = 0
        file = open(file_path,"r")
        for line in file.readlines():
            if "adaptation count" in line:
                words = line.split(" ")
                time = words[-1]
                count = int(words[-2])
                if int(count) > prev_count:
                    adaptation_count+=1
                    prev_count = count
                #print (time,count)
        print (file_path.split(".")[0] + " " + str(adaptation_count))

        bar_y_list.append(adaptation_count)

    bar_point_list = [0.42,0.41,0.42,0.45,0.25,0.48,0.30,0.35,0.47,0.30,0.48,0.45]
    bar_x_list=("Reactive","Pro_h5st","Pro_h5mt","Pro_h5lt","Pro_h10st","Pro_h10mt","Pro_h10lt","Pro_h15st","Pro_h15mt",
                       "Pro_h15lt","Pro_h30st","Pro_h30mt","Pro_h30lt",)

    bar_y_list.insert(0,762) # Reactive number of adaptations
    bar_point_list.insert(0,0.58) # Reactive number of adaptations
    '''
    print (bar_y_list)
    print (bar_x_list)

    y_pos = np.arange(len(bar_x_list))
    
    plt.bar(y_pos,bar_y_list, align='center', alpha=0.5)
    plt.xticks(y_pos, bar_x_list,rotation=40, ha='right')
    plt.ylabel('# of adaptations')
    plt.xlabel('Approaches')
    plt.title('Adaptation Count of Different Approaches')
    plt.show()
    '''

    # set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'
    plt.rcParams['text.color'] = '#333F4B'

    # create some fake data
    percentages = pd.Series(bar_point_list,
                            index=bar_x_list)
    df = pd.DataFrame({'percentage': percentages})
    #df = df.sort_values(by='percentage')

    # we first need a numeric placeholder for the y axis
    my_range = list(range(1, len(df.index) + 1))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # create for each expense type an horizontal line that starts at x = 0 with the length
    # represented by the specific expense percentage value.
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#ED6A5A', alpha=0.7, linewidth=5)

    # create for each expense type a dot at the level of the expense percentage value
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#ED6A5A', alpha=0.9)

    # set labels
    ax.set_xlabel('Adaptation Count Ratio', fontsize=15, fontweight='black', color='#333F4B')
    ax.set_ylabel('')

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)

    # add an horizonal label for the y axis
    #fig.text(0.0, 0.96, 'Approach', fontsize=15, fontweight='black', color='#333F4B')

    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    # set the spines position
    plt.tight_layout()
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))
    plt.savefig("./plots/ada_count_bar_graph.png", dpi=300)

    #plt.show()

if __name__ == '__main__':
    #result_generator("./data/" + "aggregate_energy_proactive.csv","noadap","./plots/")
    #cumulative_plot_generator()
    #box_plot_generator()
    #multi_box_plot_generator()
    #box_plot_test()
    #tester()
    log_analyzer()