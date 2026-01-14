# YSI
This code is meant to process raw csv files from YSI analyzers. 
The test_YSI.csv file is an example of what the code uses. The only required columns are the Plate Name, Well ID, Chemistry, and Concentration (although these can be under different names)

The code uses C1V1 = C2V2 = C3V3 to account for evaporation over time and calls C1 pre-evap, C2 baseline-evap, and C3 final-evap wells. These plates are controls without cells.
The calibration wells are optional, and the calibration factor can be input manually if it's already known.

Finally, the concentrations can use any number of biological replicates and automatically take the average of the technical replicates from the machine. 
