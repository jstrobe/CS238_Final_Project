############################################
########Final Project Data Generation#######
############################################
################File 1######################

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

########Variables########
#########################

# Identification number
# Age
# Estimated income level on a scale of 1 - 10
# Previous donor level (0 - Has never donated; 1 - Has made isolated donations; 2 - Recurring Donations)
# Current donor level (0 - Has never donated; 1 - Has made isolated donations; 2 - Recurring Donations)
# Amount of last donation
# Amount of current donation
# Method of contact (Scale 1 - 8; Described below)


###Each time step is 3 months

###Hyper-parameters for size -- years of data y, # of people n
###5 years of data -- 20 data points for each person
###start with ~20,000 generations and test time = 1000 people
n = 1000

######Variable Intitialization#######
#####################################

###Generate random n ID's -- each a map of ID to (Age, Income)

###Generate random age between 20 and 85 -- uniform

###Income -- divide age by 10 -- randomly select from skewed bell curve with this as mean

###Previous donor level -- 0, p = 0.4; 1, p = 0.3; 2, p = 0.3

###Amount of last donation (doesn't account for possible diff. in singular vs recurring)
#######-- if prev. level = 0 -- 0;  if prev. level = 1 or 2 -- 
####### choose 50 if income < 5; choose 500 4 <income < 10; else choose 5000 


########Dynamics Model -- (donor importance not yet implemented in generation)#########
#incorporates donor importance#
#importance 1,3 likes contacts 1,2,3,4,5
#importance 2 likes contacts 6,7,8
###############################

####Contact 1, 2, 3, 4, 5####
####if donor imp < (2/3)avg 
######if level = 0
########stay with p = 0.3
########go to 1 with p = 0.5 (donate 50,p = .9; donate 500, p = .1)
########go to 2 with p = 0.2 (donate 50,p = 1)
######if level 1
#########stay with p = 0.7 (donate 0, p = 0.5, donate same, p = 0.5)
#########go to 2 with p = 0.3 (donate 50, p=0.9; donate 500, p = .1)
######if level 2
#########stay with p = 0.8 ((donate 50, p=0.9; donate 500, p = .1))
#########go to 1 with p = 0.2 (donate 0, p = 0.5; donate 50, p = 0.5)

####if donor imp < (4/3)avg 
######if level = 0
########stay with p = 0.3
########go to 1 with p = 0.5 (donate 50,p = .3; donate 500, p=.6; donate 5000, p = .1)
########go to 2 with p = 0.2 (donate 50,p = .5, donate 500, p=.5)
######if level 1
#########stay with p = 0.7 (donate 0, p = 0.5, donate same, p = 0.5)
#########go to 2 with p = 0.3 (donate 50, p=0.9; donate 500, p = .1)
######if level 2
#########stay with p = 0.7 ((donate 50, p=0.8; donate 500, p = .2))
#########go to 1 with p = 0.3 (donate 0, p = 0.5; donate 50, p = 0.5)

####if donor imp > (4/3)avg 
######if level = 0
########stay with p = 0.1
########go to 1 with p = 0.5 (donate 50,p = .2; donate 500, p = .6, donate 5000, p = .2)
########go to 2 with p = 0.4 (donate 50,p = .4; donate 500,p = 0.5, donate 5000, p = .1)
######if level 1
#########stay with p = 0.6 (donate 0, p = 0.5, donate same, p = 0.5)
#########go to 2 with p = 0.4 (donate 50, p=0.2; donate 500, p = .7; donate 5000, p = .1)
######if level 2
#########stay with p = 0.9 ((donate 50, p=0.9; donate 500, p = .1))
#########go to 1 with p = 0.1 (donate 0, p = 0.5; donate 50, p = 0.3; donate 500, p = 0.2)

####Contact 6, 7, 8####
##....see below

#######Methods of Contact (Scale 1-8)######
#####may be more ideal for training of dynamics model to do uniform, but no realistic
##For each donor level, follow the subsequent distribution; 
###level 0 --- [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]
###level 1 or 2 --- [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.08, 0.02]


###Map ID to (Age,Income)
def pdf(x):
    return 1/sqrt(2*pi) * exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2))) / 2

def skew(x,e=0,w=1,a=0):
    t = (x-e) / w
    return 2 / w * pdf(t) * cdf(a*t)

def makeIdDict(n = 1000):
    idDict = {}
    for i in range(1, n+1):
        nextId = i
        ####Generate age####
        age = np.random.randint(20, 85)
        e = float(age)/10 # mean
        w = 2.0 # scale
        x = linspace(0,10,10) 
        prob = skew(x,e,w,0)
        prob /= float(np.sum(prob))
        incomeSpace = linspace(1,10,10)
        income = int(np.asscalar(np.random.choice(incomeSpace, 1, p = prob)))
        # print age
        # print income
        # # plot(x,prob)
        # # show()
        idDict[nextId] = (age, income)
    return idDict

idDict = makeIdDict()
averageImp = 0
###for implementing more directed dynamics model
for key in idDict:
    averageImp = 0.1*idDict[key][0] + 9*idDict[key][1]
averageImp /= len(idDict)

######if level = 0
########stay with p = 0.3
########go to 1 with p = 0.5 (donate 50,p = .9; donate 500, p = .1)
########go to 2 with p = 0.2 (donate 50,p = 1)
######if level 1
#########stay with p = 0.7 (donate 0, p = 0.5, donate same, p = 0.5)
#########go to 2 with p = 0.3 (donate 50, p=0.9; donate 500, p = .1)
######if level 2
#########stay with p = 0.8 ((donate 50, p=0.9; donate 500, p = .1))
#########go to 1 with p = 0.2 (donate 0, p = 0.5; donate 50, p = 0.5)

###currently is not change given donor importance -- initialized values are different
##currently does not rely on previous donation
def nextLevelDon(prevLevel, prevDon, contact):
    nextLevel = prevLevel
    nextDon = prevDon
    
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

##id, age, income, prevLevel, prevDonation, currentLevel, curretDonation, contact
def generate(n, idDict, averageImp, y, filename):
    contactLevel0 = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]
    contactLevel12 = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.08, 0.02]
    methods = [1, 2, 3, 4, 5, 6, 7, 8]
    
    with open(filename, mode='w') as myFile:
        file_writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(1, n+1):
            lastLine = []
            lastLine.append(i)
            lastLine.append(idDict[i][0])
            lastLine.append(idDict[i][1])
            
            prevLevel = int(np.asscalar(np.random.choice([0,1,2],1,p=[0.4,0.3,0.3])))
            prevDon = 0
            if prevLevel > 0:
                if idDict[i][1] < 5:
                    prevDon = 50
                elif idDict[i][1] < 10:
                    prevDon = 500
                else:
                    prevDon = 5000
            lastLine.append(prevLevel)
            lastLine.append(prevDon)
            contact = 0
            if prevLevel == 0:
                contact = int(np.asscalar(np.random.choice(methods, 1, contactLevel0)))
            else:
                contact = int(np.asscalar(np.random.choice(methods, 1, contactLevel12)))
            
            nextLevel, nextDon = nextLevelDon(prevLevel, prevDon, contact)
            lastLine.append(nextLevel)
            lastLine.append(nextDon)
            lastLine.append(contact)
            file_writer.writerow(lastLine)
            
            for x in range(0, y*4 - 1):
                nextLine = []
                nextLine.append(i)
                nextLine.append(idDict[i][0])
                nextLine.append(idDict[i][1])
                prevLevel = lastLine[5]
                prevDon = lastLine[6]
                nextLine.append(prevLevel)
                nextLine.append(prevDon)
                
                if prevLevel == 0:
                    contact = int(np.asscalar(np.random.choice(methods, 1, contactLevel0)))
                else:
                    contact = int(np.asscalar(np.random.choice(methods, 1, contactLevel12)))
                    
                nextLevel, nextDon = nextLevelDon(prevLevel, prevDon, contact)
                nextLine.append(nextLevel)
                nextLine.append(nextDon)
                nextLine.append(contact)
                file_writer.writerow(nextLine)
                lastLine = nextLine

    pass

generate(n, idDict, averageImp, 5, 'unProcessed.csv')
