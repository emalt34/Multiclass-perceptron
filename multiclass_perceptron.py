# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:51:56 2019

@author: Emilie

"""

import xlrd 
import numpy as np
import random


#np.set_printoptions(threshold=np.nan)

def getTrainData():
    loc = ('trainSeeds.xls') 
      
    # To open Workbook 
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    
    #number of rows  
    numRows = sheet.nrows
    
    #put data in 2D list
    trainData = []
    for i in range(numRows):
        trainData.append(sheet.row_values(i))
   
    return trainData

def getTestData():
    
    # To open Workbook 
    wb = xlrd.open_workbook('testSeeds.xls') 
    sheet = wb.sheet_by_index(0) 
    
    #number of rows  
    rows = sheet.nrows
    
    #put data in 2D list
    testData1 = []
    for j in range(rows):
        testData1.append(sheet.row_values(j))
    #make testData numpy array
    testData = np.array(testData1) 
  
    return testData

    
def makeWeight(n):
    weights = np.zeros(n)
    for i in range(n):
        randFloat = random.uniform(-1,1)
        weights[i] = "%.2f" % randFloat
    return weights; 

def train():
    
    #get training data
    trainData1 = getTrainData()
    #shuffle data
    random.shuffle(trainData1) 
    #convert to numpy array
    trainData = np.array(trainData1)    
    #print(trainData)
    
    #define learning rate
    c=1
    #make the 3 initial weight vectors 
    #wtVs are nparrays 
    wtV0 = makeWeight(7)
    wtV1 = makeWeight(7)
    wtV2 = makeWeight(7)
    weightVs = [wtV0, wtV1, wtV2]    
    
    #do 300 iterations:
    for i in range(300):
        
        for i in range(len(trainData)):
            # current x values
            xi = trainData[i][:7]
            #the true class of the data (-1 is to match up with 0 index)
            trueClass = int(trainData[i][7]-1)
            #calculate activation function for each class
            y0 = np.dot(weightVs[0], xi)
            y1 = np.dot(weightVs[1], xi)
            y2 = np.dot(weightVs[2], xi)
            ys = [y0,y1,y2]
            #calculate maximum 
            maxi = max(ys)
            #if the true class does not correspond to predicted class
            if ys[trueClass] != maxi:
                #increase weights on correct class
                weightVs[trueClass] = weightVs[trueClass] + c*xi
                #decrease weights on incorrectly identified class
                for j in range(len(ys)):
                    if ys[j] == maxi:
                        weightVs[j] = weightVs[j] - c*xi
           
    return(weightVs)
        
    
def test(testData, weights):
    
    numRight=0
    #classify
    for i in range(len(testData)):
        xi = testData[i][:7]
        trueClass = int(testData[i][7]-1)
        #calculate activation function for each class
        y0 = np.dot(weights[0], xi)
        y1 = np.dot(weights[1], xi)
        y2 = np.dot(weights[2], xi)
        ys = [y0,y1,y2]
        #calculate maximum 
        maxi = max(ys)
        if ys[trueClass] == maxi:
            numRight+=1
                           
    percent = (numRight/len(testData))*100
    
    return percent

    
def runAllDataPoints():
    trainData = getTrainData()
    trainData = np.array(trainData)
    testData = getTestData()
    weights = train()
    percTrain = test(trainData, weights)
    percTest = test(testData, weights)
    print("testing data accuracy", percTrain)
    print("training data accuracy", percTest)
  

runAllDataPoints()
      

        
    







