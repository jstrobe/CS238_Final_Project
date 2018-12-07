############################################
########Final Project Policy Learning#######
############################################
################File 3######################

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

import pandas as pd
import numpy as np
    
def write_policy(qArray, filename):
    ##make sure this does this in order
    policy = np.argmax(qArray, axis = 1)
    with open(filename, 'w') as f:
        for i in range(0, qArray.shape[0]):
            #print np.asscalar(policy[i])
            f.write("{}\n".format(np.asscalar(policy[i])+1))
            

##State space size = 9

##Action space size = 8
            
#With real data, we would want to analyze the structural 
#tendencies of each MDP and preinitialize Q values
def qLearn(filename, learnRate, discount, outputFile):
    data = pd.read_csv(filename)
    datArray = data.values
    qArray = np.zeros((9,8))
    startState = np.random.randint(0,10)
    startRow = 0
    for i in range(0, datArray.shape[0]):
        if datArray[i][0] == startState:
            startRow = i
            break

    for i in range(0, 1000*datArray.shape[0]):
        action = datArray[startRow][1]-1
        reward = datArray[startRow][2]
        nextState = datArray[startRow][3]-1
        maxA = np.argmax(qArray, axis = 1)[nextState]
        qArray[startState-1][action] += learnRate*(reward+discount*(qArray[nextState][maxA]) - qArray[startState-1][action])
        startRow += 1
        if startRow == datArray.shape[0]: startRow = 0
        ###make sure to account for jumps in data
        startState = datArray[startRow][0]
    write_policy(qArray, outputFile)
    ##maybe cut-off for convergence
    
def main():
    ###Loop through all three files and create a policy for each
    inputFile = './MDP_low.csv'
    outputFile = './lowPriority.policy'
    qLearn(inputFile, 0.001,0.85,outputFile)
    
    inputFile = './MDP_medium.csv'
    outputFile = './mediumPriority.policy'
    qLearn(inputFile, 0.001,0.85,outputFile)
    
    inputFile = './MDP_high.csv'
    outputFile = './highPriority.policy'
    qLearn(inputFile, 0.001,0.85,outputFile)
    
main()
    