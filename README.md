# IoTArchML
A Machine-learning driven proactive approach for architecting self-adaptive energy efficient IoT Systems

## Installation Requirements

1. Install the latest version of JAVA - https://www.java.com/en/download/

1. Install Apache Kafka  - https://kafka.apache.org/quickstart

2. Install kafka-python - https://pypi.org/project/kafka-python/

3. Install Keras in Python -https://keras.io

4. Install Apache Spark - https://intellipaat.com/tutorial/spark-tutorial/downloading-spark-and-getting-started/

5. Install PySpark - https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0


## Project Description

1. *CupCarbon-master_4.0* contains the modified source code of cupcarbon. The application can be started by running cupcarbon.java by using IDE like [IntelliJ IDEA](https://www.jetbrains.com/idea/) or (Eclipse IDE)[https://www.eclipse.org/ide/]
2. *NdR_CO* contains the cupCarbon project of the case study mentioned in the paper. It can be opened by opening the NdR_Cup.cup filefrom the open project option available in the cupCarbon UI. Further details can be found in www.cupcarbon.com
4. The *data* folder contains the different datasets used for experimentation and evaluations
5. The *model* folder contains the machine learning models developed using Keras for predicing the energy consumption. This acts as the model store containing the various machine learning models generated for different lag values and horizons
6. The *logs* folder contains the execution logs generated during the execution of the approach. The 
6. *settings.conf* contains the inital configurations required for all programs and this is inturn read and processed by Initalizer.py
7. To test the accuracy of each of the time-series forecasting approach, execute the corresponding python file. For instance the following code allows one to test the simple naive approach for forecasting:
<pre>
python Naive_1.py
</pre>
Similar to above any other forecasting approaches can be tested using the corresponding python file

8. *plots* folder contains the RMSE and results plots generated while running each of the approaches
9. For LSTM approach, we also have a google colab notebook which was used for training the neural network. It can be found [here](https://colab.research.google.com/drive/1BS3f-bsqKE9jPpxsFvHb5Q66XZ69H5p_?usp=sharing)
10. *Adaptation_Planner.py* implements the algorithm for performing the adaptation depending on the type of adaptation specified in the settings.conf file under the section "adaptation"
11. *CupCarbon_Energy_Streamer.py* is used for streaming the energy consumption logs to the Kafka broker (performs the role of Data Streamer)
12. *Custom_Logger.py* is an extended version of logging functionality provided by Python. It allows to keep track of the execution logs
13. *Initializer.py* is a class which initalizes the configurations from settings.conf. The object of this class is used by other files for reading configurations
14. *analyze_adaptation.py* is responsible for consuming the energy consumption logs from the kafka broker. Further it uses the ML models as defined in the settings.conf to forecast the expected energy consumption of the sensor components. It is then responsible for invoking the Adaptation_Planner. For the ease of simplicity we have added the kafka version here although the main version was implemented using Spark.
15. *result_generator.py* is used to analyze the results produced by the different approaches and to generate the different types of plots as represented in the paper.

Note: To check how to create topics and conusme topics from Kafka, refer to this [repository](https://github.com/karthikv1392/SoftwareArchitecture). Further the basic kafka commands can be found [here](https://github.com/karthikv1392/IoTArchML/blob/master/kafka_commands.md)


## Execution Instructions 



