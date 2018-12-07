############################################
######Final Project Policy Performance######
############################################
################File 4######################

# Steps: 
# 1. File 1: Generate data or incorporate real data
#     -- real data may have extra steps
#     -- Output: CSV
# 2. File 2: Pre-process data 
#     -- Generate three files -- one for each MDP
#     -- Input: raw data generated (csv)
#     -- Output: 3 CSV's
# 3. File 3: Find optimal policy for each file
#     -- Input: Processed data files
#     -- Output: Three optimal policy text files
# 4. File 4: Test each optimal policy with x random policies to compare efficiency
#     -- Input: Processed CSV's; Optimal policy files
#     -- Prints scores

import numpy as np
from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf
from pylab import plot,show
import csv
import pandas as pd

###Simulate next level and donation given policy and state
def nextLevelDon(state, contact):
    ####state = 3*Donor Level + Donor Magnitude - 1
    stateOne = state - 1
    prevDon = stateOne % 3
    prevLevel = (stateOne - prevDon) / 3
    
    if contact < 6:
        if prevLevel == 0:
            nextLevel = int(np.asscalar(np.random.choice([0,1,2],1,p=[0.3,0.5,0.2])))
            nextDon = int(np.asscalar(np.random.choice([50,500,5000],1,p=[0.6,0.3,0.1])))
        elif prevLevel == 1:
            nextLevel = int(np.asscalar(np.random.choice([1,2],1,p=[0.7,0.3])))
            if nextLevel == 1:
                nextDon = int(np.asscalar(np.random.choice([0, 50,500,5000],1,p=[0.5,0.3,0.15,0.05])))
            else:
                nextDon = int(np.asscalar(np.random.choice([50, 500, 5000],1,p=[0.4,0.55,0.05])))
        elif prevLevel == 2:
            nextLevel = int(np.asscalar(np.random.choice([1,2],1,p=[0.2,0.8])))
            if nextLevel == 1:
                nextDon = int(np.asscalar(np.random.choice([0, 50, 500],1,p=[0.55,0.4,0.05])))
            else:
                nextDon = int(np.asscalar(np.random.choice([50, 500, 5000],1,p=[0.4,0.55,0.05])))
    else:
        if prevLevel == 1:
            nextLevel = int(np.asscalar(np.random.choice([1,2],1,p=[0.6,0.4])))
            if nextLevel == 1:
                nextDon = int(np.asscalar(np.random.choice([0, 50,500,5000],1,p=[0.5,0.3,0.15,0.05])))
            else:
                nextDon = int(np.asscalar(np.random.choice([50, 500, 5000],1,p=[0.3,0.6,0.1])))
        elif prevLevel == 2:
            nextLevel = int(np.asscalar(np.random.choice([1,2],1,p=[0.1,0.9])))
            if nextLevel == 1:
                nextDon = int(np.asscalar(np.random.choice([0, 50, 500],1,p=[0.55,0.4,0.05])))
            else:
                nextDon = int(np.asscalar(np.random.choice([50, 500, 5000],1,p=[0.35,0.6,0.05])))
        
    return nextLevel, nextDon

def donMagnitude(n):
    if n < 100:
        return 0
    if n < 1000:
        return 1
    if n > 999:
        return 2

def calculateScores():
    ########Generate Random Policies########
    ########################################
    with open('./randLow.policy', 'w') as f:
        ##state space size
        for i in range(0, 9):
            f.write("{}\n".format(np.random.randint(1,9)))

    with open('./randMedium.policy', 'w') as f2:
        ##state space size
        for i in range(0, 9):
            f2.write("{}\n".format(np.random.randint(1,9)))

    with open('./randHigh.policy', 'w') as f3:
        ##state space size
        for i in range(0, 9):
            f3.write("{}\n".format(np.random.randint(1,9)))

    ######Test Policies in "Simulator"######
    ########################################

    ########Low Importance Donors###########
    #######################################

    ###Get average state for initialization
    one = pd.read_csv('./MDP_low.csv')
    datOne = one.values
    aveStateOne = 0
    for i in range(0, datOne.shape[0]):
        aveStateOne += datOne[i][0]
    aveStateOne = int(aveStateOne/datOne.shape[0])

    ###load policy -- low importance
    polOne = []
    polOneF = open('./lowPriority.policy', 'r') 
    for line in polOneF:
        polOne.append(int(line))

    ###Calculate utility for policy
    rewardOne = 0
    stateOne = aveStateOne
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polOne[stateOne - 1]
        nextLevel, nextDon = nextLevelDon(stateOne, contact)
        rewardOne += nextDon
        stateOne = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardOne = float(rewardOne)/1000

    ####Test Random low policy#####
    ###load random policy -- low importance
    polOneRand = []
    polOneRF = open('./randLow.policy', 'r') 
    for line in polOneRF:
        polOneRand.append(int(line))

    ###Calculate utility for random policy
    rewardOneRand = 0
    stateOneRand = aveStateOne
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polOneRand[stateOneRand - 1]
        nextLevel, nextDon = nextLevelDon(stateOneRand, contact)
        rewardOneRand += nextDon
        stateOneRand = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardOneRand = float(rewardOneRand)/1000

    ###Calculate comparison score
    scoreOne = rewardOne - rewardOneRand
    #print "Score for Low Potential Donor Policy: ", scoreOne 

    #######Medium Importance Donors#########
    #######################################

    ###Get average state for initialization
    two = pd.read_csv('./MDP_medium.csv')
    datTwo = two.values
    aveStateTwo = 0
    for i in range(0, datTwo.shape[0]):
        aveStateTwo += datTwo[i][0]
    aveStateTwo = int(aveStateTwo/datTwo.shape[0])

    ###load policy -- medium importance
    polTwo = []
    polTwoF = open('./mediumPriority.policy', 'r') 
    for line in polTwoF:
        polTwo.append(int(line))

    ###Calculate utility for policy
    rewardTwo = 0
    stateTwo = aveStateTwo
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polTwo[stateTwo - 1]
        nextLevel, nextDon = nextLevelDon(stateTwo, contact)
        rewardTwo += nextDon
        stateTwo = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardTwo = float(rewardTwo)/1000

    ####Test Random medium policy#####
    ###load random policy -- medium importance
    polTwoRand = []
    polTwoRF = open('./randMedium.policy', 'r') 
    for line in polTwoRF:
        polTwoRand.append(int(line))
    rewardTwoRand = 0

    ###Calculate utility for random policy
    rewardTwoRand = 0
    stateTwoRand = aveStateTwo
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polTwoRand[stateTwoRand - 1]
        nextLevel, nextDon = nextLevelDon(stateTwoRand, contact)
        rewardOneRand += nextDon
        stateTwoRand = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardTwoRand = float(rewardTwoRand)/1000

    ###Calculate comparison score
    scoreTwo = rewardTwo - rewardTwoRand
    #print "Score for Medium Potential Donor Policy: ", scoreTwo 

    ########High Importance Donors###########
    #######################################

    ###Get average state for initialization
    three = pd.read_csv('./MDP_low.csv')
    datThree = three.values
    aveStateThree = 0
    for i in range(0, datThree.shape[0]):
        aveStateThree += datThree[i][0]
    aveStateThree = int(aveStateThree/datThree.shape[0])

    ###load policy -- high importance
    polThree = []
    polThreeF = open('./highPriority.policy', 'r') 
    for line in polThreeF:
        polThree.append(int(line))

    ###Calculate utility for policy
    rewardThree = 0
    stateThree = aveStateThree
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polThree[stateThree - 1]
        nextLevel, nextDon = nextLevelDon(stateThree, contact)
        rewardThree += nextDon
        stateThree = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardThree = float(rewardThree)/1000

    ####Test Random high policy#####
    ###load random policy -- high importance
    polThreeRand = []
    polThreeRF = open('./randHigh.policy', 'r') 
    for line in polThreeRF:
        polThreeRand.append(int(line))
    rewardThreeRand = 0

    ###Calculate utility for random policy
    rewardThreeRand = 0
    stateThreeRand = aveStateThree
    nextLevel = 0
    nextDon = 0
    for i in range(0, 1000):
        contact = polThreeRand[stateThreeRand - 1]
        nextLevel, nextDon = nextLevelDon(stateThreeRand, contact)
        rewardThreeRand += nextDon
        stateThreeRand = 3*nextLevel + donMagnitude(nextDon) + 1

    rewardThreeRand = float(rewardThreeRand)/1000

    ###Calculate comparison score
    scoreThree = rewardThree - rewardThreeRand
    #print "Score for High Potential Donor Policy: ", scoreThree
    
    return scoreOne, scoreTwo, scoreThree

###Get average scores for 100 iterations
scoreOneAve = 0
scoreTwoAve = 0
scoreThreeAve = 0
for i in range(0, 100):
    scoreOne, scoreTwo, scoreThree = calculateScores()
    scoreOneAve += scoreOne
    scoreTwoAve += scoreTwo
    scoreThreeAve += scoreThree
    
scoreOneAve = float(scoreOneAve) / 100
scoreTwoAve = float(scoreTwoAve) / 100
scoreThreeAve = float(scoreThreeAve) / 100

print "Average Score for Policy One: ", scoreOneAve
print "Average Score for Policy Two: ", scoreTwoAve
print "Average Score for Policy Three: ", scoreThreeAve
