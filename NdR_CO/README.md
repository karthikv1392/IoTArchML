## NdR Project modeled in CupCarbon

1. Load the project into CupCarbon by selecting the NdR_CO.cup file from the open project option in CupCarbon
2. *scripts* folder contains the SenScript code for each sensor in the system
3. *config* folder contains the hardware configuration of the sensors nodes
4. *results* folder initially will be empty. During the simulation the energy consumption log file will be created here with the name *wisen_simulation.csv*
5. *logs* folder initally will be empty. Duiring the simulation the logs of different events that happens in the system including the data sent from one sensor node to another can be obtained from here
6. *natevents* folder contains the event file for each of the sensor ndoes. These basically denotes the data that will be used for simulation by the respective sensor nodes. These are created based on the data observed during the actual NdR event
