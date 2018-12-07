############################################
######Final Project Data Preprocessing######
############################################
################File 2######################

# Steps: 
# 1. File 1: Generate data or incorporate real data
#     -- real data may have extra steps
#     -- Output: CSV
# 2. File 2: Pre-process data 
#     -- Generate three files -- one for each MDP
#     -- Input: raw data generated (csv)
#     -- Output: 3 CSV's
# 3. Find optimal policy for each file
#     -- Input: Processed data files
#     -- Output: Three optimal policy text files
# 4. Test each optimal policy with x random policies to compare efficiency
#     -- Input: Processed CSV's; Optimal policy files
#     -- Prints scores

import csv
import pandas as pd
import numpy as np

data = pd.read_csv('./unProcessed.csv', header=None)
datArray = data.values

###First we loop through the data once to find 
###the average "Donor Potential", or importance, 
###according to the following formula:

###Donor Potential= 0.1∗Age+ 9∗(Estimated Income)

averagePotential = 0

for i in range(0, datArray.shape[0]):
    averagePotential += 0.1*datArray[i][1] + 9*datArray[i][2]
averagePotential = averagePotential / datArray.shape[0]

######Re-Format Data Into 3 Files#####
######################################

####Here, we want to categorize our data points
####and write them into a new file in the four-tuple
####format, including current state, action, reward,
####and next state. States and actions will be 1-indexed
####for use in algorithm implementation

####state is defined by the following formula:

####State = 3∗(Donor level) + Donation Magnitude + 1

def donMagnitude(n):
    if n < 100:
        return 0
    if n < 1000:
        return 1
    if n > 999:
        return 2
    
##File format of raw data
##id, age, income, prevLevel, prevDonation, currentLevel, curretDonation, contact

##File format of processed data
##state, action, reward, next state

###file for lowest importance donors, beneath threshold 1
###Threshold 1 = (2/3) * Average Donor Potential
with open("MDP_low.csv", mode='w') as myFile:
    file_writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['s', 'a', 'r', 'sp'])
    for i in range(0, datArray.shape[0]):
        if (0.1*datArray[i][1] + 9*datArray[i][2]) < (float(2)/3)*averagePotential:
            nextLine = []
            lastState = 3*datArray[i][3] + donMagnitude(datArray[i][4]) + 1
            nextState = 3*datArray[i][5] + donMagnitude(datArray[i][6]) + 1
            nextLine.append(lastState)
            nextLine.append(datArray[i][7])
            nextLine.append(datArray[i][6])
            nextLine.append(nextState)
            file_writer.writerow(nextLine)

###file for medium importance donors, between threshold 1 and 2
###Threshold 1 = (2/3) * Average Donor Potential
###Threshold 2 = (4/3) * Average Donor Potential
with open("MDP_medium.csv", mode='w') as myFileTwo:
    file_writerTwo = csv.writer(myFileTwo, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writerTwo.writerow(['s', 'a', 'r', 'sp'])
    for i in range(0, datArray.shape[0]):
        if (0.1*datArray[i][1] + 9*datArray[i][2]) >= (float(2)/3)*averagePotential and (0.1*datArray[i][1] + 9*datArray[i][2]) < (float(4)/3)*averagePotential:
            nextLine = []
            lastState = 3*datArray[i][3] + donMagnitude(datArray[i][4]) + 1
            nextState = 3*datArray[i][5] + donMagnitude(datArray[i][6]) + 1
            nextLine.append(lastState)
            nextLine.append(datArray[i][7])
            nextLine.append(datArray[i][6])
            nextLine.append(nextState)
            file_writerTwo.writerow(nextLine)

###file for high importance donors, above threshold 2
###Threshold 2 = (4/3) * Average Donor Potential
with open("MDP_high.csv", mode='w') as myFileThree:
    file_writerThree = csv.writer(myFileThree, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writerThree.writerow(['s', 'a', 'r', 'sp'])
    for i in range(0, datArray.shape[0]):
        if (0.1*datArray[i][1] + 9*datArray[i][2]) >= (float(4)/3)*averagePotential:
            nextLine = []
            lastState = 3*datArray[i][3] + donMagnitude(datArray[i][4]) + 1
            nextState = 3*datArray[i][5] + donMagnitude(datArray[i][6]) + 1
            nextLine.append(lastState)
            nextLine.append(datArray[i][7])
            nextLine.append(datArray[i][6])
            nextLine.append(nextState)
            file_writerThree.writerow(nextLine)