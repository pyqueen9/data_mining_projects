# Neural network 
# Data Mining

from scipy.io import arff
import numpy as np
import math
import copy
import argparse
import sys
from os import path
import os
import random  
import time

# Hyper parameters to tune
L_RATE = 0.1
HIDDEN_NODES = 20
EPOCHS_TOTAL = 10

# Parse file to clean and store data 
def parseFile():
    for line in file:
        if line.startswith("@attribute"):
            line=line.strip("\n")
            line = line.replace("," , "")
            list = line.split(" ")
            if ('{') in list:
                list.remove('{')
            if ('}') in list:
                list.remove('}')
            list.remove("@attribute")
            list.pop(0)  
            atts.append(list )  
        if line[0].isdigit():
            line=line.strip("\n")
            line = line.replace("," , "")
            list= line.split(" ")
            instances.append(list)

#  K - fold cross validation
def kFoldCV():
    # for k folds create testSet & trainSet holding each round of training and testing data
    for i in range(k):
        testSet.append( instances[i*fold:][:fold]) 
        trainSet.append(instances[:i*fold] + instances[(i+1)*fold:])

# Helper function to calculate the nominal att modes to replace missing values
def calculateNominalAttModes():
    index =0
    count =0 
    for j in range(len(atts)):
        newarr = [0]*len(atts[j])
        nominalAttCounts.append(newarr)
    for i in range(len(instances)):  
        for k in range(len(instances[i])): 
            if atts[k][0] != "numeric": 
                if instances[i][k] != "?":
                    nominalAttCounts[k][(atts[k].index(instances[i][k]))] += 1 # increment counters +1 for each instance of value
    for j in range(len(atts)):
        index = nominalAttCounts[j].index(max(nominalAttCounts[j])) # get the index of the mode att
        if atts[j][0] != "numeric":  
            atts[j].append(index) # add index of mode to that att in atts array @ last position

# Helper function to calculate the numeric means to replace missing values
def calculateNumericAttMeans():
    sum = 0
    total = len(instances)
    avg =0 
    for i in range(len(atts)): # for all the attributes
        numericMean = []
        if atts[i][0] == "numeric":
            for k in range(len(instances)): 
                if instances[k][i] != "?":   
                    numericMean.append(int(instances[k][i]))          
            numericMean= np.asarray(numericMean)
            # 2nd to last element of numeric att array will be mean
            mean = numericMean.mean()
            atts[i].append(mean)   
            stdev = numericMean.std()
            atts[i].append(stdev)           
# Replace missing values in the data set
def replaceMissing():
    for i in range(len(instances)): 
        for k in range(len(instances[i])):  
            if (instances[i][k] == "?"):
                if atts[k][0] == "numeric":
                    instances[i][k] = atts[k][1]
                else:
                    modeIndex = atts[k][-1]
                    instances[i][k] = atts[k][modeIndex]
# Normalize all the numeric attributes with Z-score
def normalizeZScore():
    for k in range(len(atts)):
        for i in range(len(instances)):
            if (atts[k][0] == "numeric"):
                instances[i][k]  = int(instances[i][k])
                instances[i][k] =(instances[i][k] - atts[k][1] )/ atts[k][2]
             
# Update the age attribute by categorizing into bins
def updateAttribute( attsValues): 
    binArr = []
    instancesValues = []
    pos = 0 # index of age attribute 
    # best splits are found with Weka tool
    bestSplits = [ 24.5, 31.5, 38.5, 46, 53]
    bestSplits = sorted(bestSplits)
    bestSplits.insert(0, 0)
    size =len(bestSplits)
    lastAttVal = attsValues[-1] +1
    # create an array with different bins 
    for i in range(size ):
        if i == size-1:
            binArr.append(list(range(last, lastAttVal )))
        else: 
            first = int( bestSplits[i]) +1
            last = int(bestSplits[i+1]) +1
            binArr.append(list(range(first,last) ))
    # Update att in instances list with bin values (does not start at 0!): [1...6]
    for i in range(len(instances)):
        instancesValues.append(int(instances[i][pos]))
    instancesValues = np.array(instancesValues) #transform to use np method
    instancesValues = np.digitize(instancesValues, bestSplits)
    for i in range(len(instances)):
        # update value of att in instances list to the bin value based on
        # which bin it falls into
        instances[i][pos] = instancesValues[i]
    atts[pos].clear()
    for i in range(len(binArr)):
         atts[pos].append(i+1)
    return instances

# Get transformed input layer for NN with all of the training tuples for this fold
def getInputs(trainData = []):
    # set up input layer: one node for each nominal att value and one node for a numeric input 
    inputLayer = []
    for i in range(len(atts)-1):
        if atts[i][0] == "numeric":
            nodes = [0]* 1
            inputLayer.append(nodes)
            inputLayer[i][0] = trainData[i]
        elif atts[i][0] != "numeric":
            if ( i ==0): # if it is the age att
                nodes = [0] * len(atts[i])
            elif ( i!= 0):
                nodes = [0] * (len(atts[i])-1)
            inputLayer.append(nodes)
            idx = atts[i].index(trainData[i])
            inputLayer[i][idx] = 1
    inputLayer = sum(inputLayer, [])
    return inputLayer

# This is a helper function to categorize the age attribute - 
# it finds an array of distinct values for the attribute
def getAttsValues(attsValues):
    pos = 0
    for i in range(len(instances)):
        attsValues.append(int(instances[i][pos]))
    unique_attValues =  set(attsValues) # change to set to filter unique values
    attsValues = list(unique_attValues)
    attsValues = sorted(attsValues) # change back to sorted array
    return attsValues
    
# Training for NN - implements Back propagation algorithm
def train(trainData = [], testSet = []):
    epochs = 0 
    outputNodes = 1
    targets = []
    inputLayer = []
    sizeAtts = 110
    np.random.seed(3123)
    # initialize hidden and output layers weights + biases
    nn.hiddenB = np.random.uniform(-0.5,0.5, size=(HIDDEN_NODES,1))    
    nn.hiddenW = np.random.uniform(-0.5,0.5, size=(sizeAtts,HIDDEN_NODES))
    nn.outputB = np.random.uniform(-0.5,0.5, size=(outputNodes,1))
    nn.outputW = np.random.uniform(-0.5,0.5, size=(HIDDEN_NODES,outputNodes))
    # transform the input layer and get the target values
    for t in range(len(trainData)):
        inputs = getInputs(trainData[t])
        I = np.array([inputs]).T
        inputLayer.append(I)
        target = atts[-1].index( trainData[t][-1])
        targets.append(target)
    
    # loop for every epoch of training - terminating condition is # epochs
    while(epochs < EPOCHS_TOTAL):
        cnt = 0
        err = []
        correct_tr =0
        total = 0
        for t in range(len(inputLayer)):
            cnt += 1
            total += 1
            I  = inputLayer[t]
            #FEED FORWARD pass 
            hiddenIn = nn.hiddenW.T @ I + nn.hiddenB
            hiddenOut = sigmoid(hiddenIn)
            outputIn = nn.outputW.T @ hiddenOut + nn.outputB
            outputOut = sigmoid(outputIn)
            # BACK PROPAGATE ERRORS
            errO=[]
            errH=[] 
            # for 1 output node
            e = outputOut[0]*(1-outputOut[0])*(targets[t] - outputOut[0])
            errO.append(e)
            errO = np.array(errO)    
            # for 2 output nodes
            '''
            for i in range(outputNodes): 
                e = outputOut[i]*(1-outputOut[i])*(targets[t]-outputOut[i])
                errO.append(e)
            errO = np.array(errO)
            '''     
            # return to hidden layer - calculate errors
            e = errO*sigmoid(outputOut,der=True)
            e = np.array(e).T
            errH = e.dot(nn.outputW.T) 
            # UPDATE weights and biases in network
            nn.hiddenW = nn.hiddenW + ((L_RATE)* (I @ errH))
            nn.outputW = nn.outputW + ((L_RATE)* (hiddenOut @ errO.T))
            delta_b =L_RATE * errH
            nn.hiddenB+= delta_b.T
            delta_b = L_RATE* errO
            nn.outputB+= delta_b
            err.append(0.5*math.pow((outputOut[0] -targets[t]),2)) 
            # check training accuracy and mean sq. errors
            if (outputOut[0] > 0.5):
                prediction = 1
            elif(outputOut[0] <=0.5):
                prediction = 0
            if (prediction == targets[t]):
                correct_tr += 1 
            if cnt % 1000 == 0:
                mean_sq_err = np.average(err)
                accuracy = correct_tr / total 
                print("mse {:.4f}, tr acc: {:.4f}".format(mean_sq_err, accuracy), end='\r') 
        if (epochs == 0):
            print( "Output error after 1st epoch: ")
            print(errO)
        epochs += 1
        print(epochs)
    print("Error at last tuple's output" )
    print(errO)
    print("Epochs #: ")
    print(epochs)
    print("Hidden nodes: ")
    print(HIDDEN_NODES)
    print("Learning rate: ")
    print(L_RATE)

# Calculate sigmoid activation function  
def sigmoid(L, der = False):
    if(der == True):
        return L*(1-L)   
    return 1 / (1 + np.exp(-L))
   
# Testing method for NN and evaluation results
def classify(testSet=[]):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg= 0
    trueNeg=0
    size= len(testSet)
    nn.correct = 0 # reset
    neg = 0 # index zero of class label
    pos = 1 # index 1 of class label
    
    for t in range(len(testSet)):
        target = atts[-1].index( testSet[t][-1])
        inputs = getInputs(testSet[t])
        I = np.array([inputs]).T
        # forward pass 
        hiddenIn = nn.hiddenW.T @ I + nn.hiddenB
        hiddenOut = sigmoid(hiddenIn)
        outputIn = nn.outputW.T @ hiddenOut + nn.outputB
        outputOut = sigmoid(outputIn)
        #  get predictions of network 
        if (outputOut[0] >0.5):
            prediction = pos
        elif(outputOut[0] <=0.5):
            prediction = neg
        if (prediction == target):
            if(prediction == 1):
                nn.correct += 1
                truePos += 1
            elif(prediction ==0):
                nn.correct += 1
                trueNeg+= 1
        trueNeg = size - truePos
               
        if(prediction != target and target == 1):
            falseNeg += 1
                  
        if(prediction != target and target == 0):
            falsePos += 1
    
    precision =  round((truePos/(truePos + falsePos))*100 ,2)
    recall = round((truePos/ (truePos + falseNeg))*100, 2)
    specificity = round((trueNeg/ (trueNeg + falsePos))*100,2)
    f1 =  round(((2*recall*precision)/(recall+precision)),2)
    
	# evaluation metrics 
    print("Precision: ")
    print(precision)
    print("Recall: ")
    print(recall)
    print("Specificity: ")
    print(specificity)
    print("F1: ")
    print(f1)
    

# MAIN  function

filename = sys.argv[-1]
cwd = os.getcwd()
path = os.path.join(cwd, filename)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", dest="filename")
	args = parser.parse_args()
    
file = open( filename)

k = 5
total = 0
atts = [] # array holds nominal att values, or 'numeric' att label, & numeric means
instances = [] # array holds all the samples/instances
trainSet= [] # array holds each round of training data
testSet= [] # holds each round of testing data
nominalAttCounts= []
attsValues = [] 
  
parseFile()
fold = len(instances) / 5
fold = round(fold)
kFoldCV()

# NEURAL NETWORK 
class NN: 
    hiddenB = []
    outputB = []
    hiddenW= []
    outputW = []
    correct =0
    total = 0
   
nn = NN()

# Pre processing - same as before
calculateNominalAttModes()
calculateNumericAttMeans()
replaceMissing()
# Categorize the age attribute
attsValues =getAttsValues(attsValues)
updateAttribute( attsValues)
# Normalization 
normalizeZScore()

# Training & Classification  
total= len(testSet[0])
for i in range(k):
    start = time.time()
    print(" MODEL 1")
    print("***********FOLD #***********") 
    print(i+1)
    print("Start training")
    train(trainSet[i], testSet[i])
    print("Finished training")
    print(" " )
    classify(testSet[4])
    print("Accuracy: ")
    acc = round((nn.correct/total)*100,2)
    print(acc)
    stop = time.time()
    print("--- %s seconds for this fold ---" % (stop - start ))