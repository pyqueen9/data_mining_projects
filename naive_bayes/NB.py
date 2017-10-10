
# Naive Bayes in Python
# Data Mining 

# Data preprocessing for dataset provided - parsing the file
def parseFile():
    for line in file:
        if line.startswith("@attribute"):
            line = line.strip("\n")
            line = line.replace("," , "")
            list = line.split(" ")
            if ('{') in list:
                list.remove('{')
            if ('}') in list:
                list.remove('}')
            list.remove("@attribute")
            list.pop(0) # remove att name
            atts.append(list )  
        if line[0].isdigit():
            line = line.strip("\n")
            line = line.replace("," , "")
            list = line.split(" ")
            instances.append(list)

# Calculate a numeric attribute mean
def calculateNumericAttMeans():
    sum, avg = 0
    total = len(instances)
    for i in range(len(atts)): # for all the attributes
        numericMean = []
        if atts[i][0] == "numeric":
            for k in range(len(instances)): 
                if instances[k][i] != "?":   # if the attribute is not missing
                    numericMean.append(int(instances[k][i]))
            numericMean= np.asarray(numericMean)
            # 2nd to last element of numeric att array will be mean
            mean = numericMean.mean()
            atts[i].append(mean)   
            
# Calculate a nominal/categorical attribute mode 
def calculateNominalAttModes():
    index , count = 0 
    for j in range(len(atts)):
        newarr = [0]*len(atts[j])
        nominalAttCounts.append(newarr)
    for i in range(len(instances)):  
        for k in range(len(instances[i])):  
            if atts[k][0] != "numeric": 
                if instances[i][k] != "?":
                    nominalAttCounts[k][(atts[k].index(instances[i][k]))] += 1  
    for j in range(len(atts)):
        index = nominalAttCounts[j].index(max(nominalAttCounts[j])) # get the index of the mode att
        atts[j].append(index) # add index of mode to that att in atts array @ last position
 
# Calculate k-fold Cross validation
def kFoldCV():
    # for k folds create testSet & trainSet holding each round of training and testing data
    for i in range(k):
        testSet.append( instances[i*fold:][:fold]) 
        trainSet.append(instances[:i*fold] + instances[(i+1)*fold:])
   
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
              
# Calculate the entropy of the class attribute 
def calculateEntropyOfClassAtt(pos):
    # last attribute is the class - ignore mode  
    attSize= len(atts[pos]) -1
    classLabelCounts= [0]*attSize
    for i in range(len(instances)):
        if (instances[i][pos] == atts[pos][0]):
            classLabelCounts[0] += 1
        else:  
            classLabelCounts[1] += 1
    total = classLabelCounts[0] + classLabelCounts[1]  
    first_cl= classLabelCounts[0]
    prob_first = first_cl/total
    second_cl = classLabelCounts[1]
    prob_second = second_cl/total 
    entropy_cl = -((prob_first *math.log(prob_first,2)) + ((   prob_second *math.log(   prob_second,2))))
    return entropy_cl
     

# Calculate information gain 
def calculateInfo(pos, binVal):
    # helper function to calculate info from 2 entropies for splits
    # returns the info value for this split
    # pos is the position of the att in atts array
    # binVal is the split value for this bin ( ex: for age 19.5)
    entropies = [] 
    attSize= 2 # binary class attribute
	# arrays for class labels
    lessCounts= [0]*attSize # less than values ; 0 = >50k and 1 = <=50k
    greaterCounts = [0]*attSize # greater than values ; 0 = >50k and 1 = <=50k
    for i in range(len(instances)):
        # less than split value and class label 0= >50k
        if (int(instances[i][pos]) <=  binVal) and (instances[i][-1] == atts[-1][0]):
            lessCounts[0] += 1 # 0 = less or equal  
        # less than split value and class label 1 <=50k
        elif (int(instances[i][pos])) <=  binVal and (instances[i][-1] == atts[-1][1]):
            lessCounts [1] += 1 
        # 1 = greater than split value and class label 0 = >50k
        elif (int(instances[i][pos]) >  binVal) and (instances[i][-1] == atts[-1][0]):
            greaterCounts[0] += 1 # 0 = less or equal  
        # greater than split value and class label 1 <=50k
        elif (int(instances[i][pos]) >  binVal) and (instances[i][-1] == atts[-1][1]):
            greaterCounts [1] += 1 # 1 = greater than

    # get entropy  for less than/equal to binVal Counts    
    totalLess = lessCounts[0] + lessCounts[1]  
    first = lessCounts[0]
    prob_first = first/totalLess
    second = lessCounts[1]
    prob_second = second/totalLess
    #check for zero probabilities -> undefn logs; if prob = zero -> entropy is 0; 
    # there is a perfect split
    if prob_first ==0  or prob_second == 0:
        entropy_firstBin = 0
    else:
    # note log base 2 second parameter
        entropy_firstBin = -((prob_first *math.log(prob_first,2)) + ((prob_second *math.log(prob_second,2))))
    
    # get entropy  for greater than counts
    totalGreater = greaterCounts[0] + greaterCounts[1]  
    first= greaterCounts[0]
    prob_first = first/totalGreater
    second = greaterCounts[1]
    prob_second = second/totalGreater 
    if prob_first ==0  or prob_second == 0:
        entropy_secondBin = 0
    else:
        entropy_secondBin = -((prob_first *math.log(prob_first,2)) + ((prob_second *math.log(   prob_second,2))))
    entropies.append(entropy_firstBin)
    entropies.append(entropy_secondBin)
    
    # Info calculation
    total = totalLess + totalGreater
    prob_lessBin = totalLess/total
    prob_greaterBin = totalGreater/total
    info = (prob_lessBin*entropies[0]) + ( prob_greaterBin*entropies[1])
    return info

# Calculate entropy discretization 
def entropyDiscretize(classEnt, pos, attsValues, left):
    # returns an array of best splits
    # classEnt = the class attribute entropy
    # pos = position of attribute to discretize
    # attsValues = list of possible values based on set of samples
    # left = flag (T/F) value if traversing LHS
    gains = []
    midValues = []
    rightArray = [] # this will be used to keep right half of original array to traverse RHS
    for k in range(len(attsValues) -1):
        first = int(attsValues[k])
        second = int(attsValues[k+1])
        mid = (first +second)/2
        midValues.append(mid)
    for m in range(len(midValues)): #for each split
        # calculates net entropy and info for this split value
        info = calculateInfo(pos, midValues[m])
        # calculates gains for this split value and saves current gain
        gains.append(calculateGain(info, classAttEntropy))
    maxGain = max(gains)
    allGains.append(maxGain)
    idx = gains.index(max(gains))
    bestSplit = midValues[idx]
    bestSplits.append(bestSplit)
    
    if maxGain > 0.05: 
        newArray = getAttsValues(4,1,bestSplit, attsValues) 
        if left == 1:
            rightArray = newArray[1]
        entropyDiscretize(classEnt, pos, newArray[0], 0)
    else: 
        return ;
    if left == 1:
        if maxGain > 0.05:
            newArray = getAttsValues(4,1,bestSplit, rightArray)  
            entropyDiscretize(classEnt, pos, rightArray,1)
        else:  
            return bestSplits
			
# Calculate gain 
def calculateGain(info, classAttEntropy):
    gain = classAttEntropy - info
    return gain;

# Helper function to get the values for an attribute 
def getAttsValues(pos, split, nextSplit, attsValues):
        # pos - specifiy att position
        # split = flag (T/F) to indicate if we are splitting the array into 2
        # nextSplit = value on which to split on
        # attsValues = an array of values to split
    if(split == 1): # partition into 2
        newArray= []
        firstHalf = []
        secondHalf =[]
        for  l in range(len(attsValues)):
            if attsValues[l] <= nextSplit:
                firstHalf.append(attsValues[l])
            else: secondHalf.append(attsValues[l])
        newArray.append(firstHalf)
        newArray.append(secondHalf)
        return newArray
    for i in range(len(instances)):
        attsValues.append(int(instances[i][pos]))
    unique_attValues =  set(attsValues)  
    attsValues = list(unique_attValues)
    attsValues = sorted(attsValues)  
    return attsValues

# Update attribute
def updateAttribute(pos, attsValues, bestSplits):
    # create bins for this attribute
    # pos = pos of att
    # attsValues = all the possible values for an att
    # bestSplits = aray of best splitting values for this att
    binArr = []
    instancesValues = []
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
    # update att in instances list with bin values (does not start at 0!): [1...6]
    for i in range(len(instances)):
        instancesValues.append(int(instances[i][pos]))
    instancesValues = np.array(instancesValues)  
    instancesValues = np.digitize(instancesValues, bestSplits)
    for i in range(len(instances)):
        # update value of att in instances list to the bin value based on
        # which bin it falls into
        instances[i][pos] = instancesValues[i]
    # finally update atts array to calculate probability
    # of the categorical version of this attribute
    atts[pos].clear()
    for i in range(len(binArr)):
         atts[pos].append(i+1)
   
    return instances

# Calculate probabilities
def calculateNBProb( trainData= []):
    # for binaryClass
    # class label at 0: >50; Class label at 1: <=50k
    # let >50 = positive class -> prob_n
    # let <=50k = negative class -> prob_p
    # get arrays of pPos and PNeg of likelihoods probabilities ( X|C) for both classes
    classLabelCounts = [0] *(len(atts[-1])-1)
    size = len(trainData)
    sizeAtts = len(atts)
    # transform this arrays to probabilities for nominal atts 
    # by dividing each att value by totalN/totalP
    pNeg = [] # prob for each attribute given negative class -0
    pPos = [] # probabilities of atts for positive class - 1
    for j in range(sizeAtts):
        if (j == pos):
            newArr = [0]*(len(atts[j]))
            pNeg.append(newArr)
        else:
            if(atts[j][0]=="numeric"):
                newArr = [0]*(4)
                pNeg.append(newArr)
            elif(atts[j][0] !="numeric"): 
                newArr = [0]*(len(atts[j])-1)
                pNeg.append(newArr)
    for j in range(sizeAtts):
        if (j == pos):
            newArr = [0]*(len(atts[j]))
            pPos.append(newArr)
        else:
            if(atts[j][0]=="numeric"):
                newArr = [0]*(4)
                pPos.append(newArr)
            elif(atts[j][0] !="numeric"): 
                newArr = [0]*(len(atts[j])-1)
                pPos.append(newArr)
    # negNum holds  - numeric values for this att, given negative class
    # posNum holds - numeric values for this att, given positive class 
    # save numeric data summaries - mean and std
    for k in range(len(atts)):
        if atts[k][0] == "numeric":
            negNum = []
            posNum = []
            for i in range(len(trainData)):
                if trainData[i][-1] == atts[-1][0]: # negative class
                    negNum.append(float(trainData[i][k]))
                elif trainData[i][-1] == atts[-1][1]:
                    posNum.append(float(trainData[i][k]))
            neg_mean = np.asarray(negNum).mean()
            pos_mean = np.asarray(posNum).mean()
            neg_std = np.asarray(negNum).std()
            pos_std = np.asarray(posNum).std()
            pNeg[k][0] = neg_mean
            pNeg[k][1] =neg_std
            pPos[k][0] =pos_mean
            pPos[k][1] =pos_std
        
    # Iterate over ALL THE ATTRIBUTES
    # for all the instances, check each attribute and get probability based on class label   
    # for each att create an array of counts which will at the end be transformed to probabilites
    # if i == sizeAtts-1 we are at the last att or class attribute
    # so save the counts to get the prior probabilities after
    for i in range(sizeAtts ): # ignore class attribute
         for n in range(len(trainData)):
        # check index  to save priors for class att
            if (i == sizeAtts-1):
                if trainData[n][-1] == atts[-1][0]:
                    classLabelCounts[0] +=1
                elif trainData[n][-1] == atts[-1][1]:
                    classLabelCounts[1] +=1
        #do negative class - categorical
            else:
                if trainData[n][-1] == atts[-1][0]:
                    if(atts[i][0]) != "numeric":
                        idx = int(atts[i].index(trainData[n][i]))
                        pNeg[i][(idx)] += 1
        #do positive class - categorical
                elif trainData[n][-1] == atts[-1][1]:
                    if(atts[i][0]) != "numeric":
                        pPos[i][(atts[i].index(trainData[n][i]))] += 1
    # calculate prior probabilities for class P(C_i)
    totalN = classLabelCounts[0]
    totalP = classLabelCounts[1]
    prob_n = classLabelCounts[0]/size
    prob_p = classLabelCounts[1]/size
    classPriors.append(prob_n)
    classPriors.append(prob_p)
    # calculate likelihoods for each attrbute and its values given each class  P(X_i|C)
    for i in range(len(atts) -1):
        for k in range(len(atts[i])-1):
            if atts[i][0] != "numeric":
                pPos[i][k] = (pPos[i][k] +0.5)/(totalP +1)
                pNeg[i][k] = (pNeg[i][k]+0.5)/(totalN +1)
    likelihoods.append(pNeg)
    likelihoods.append(pPos)
     
# Perform NB classification on test set 
def classifyNB(testData= []):
 
    #int label # 0 or 1 class 
    trueLabels = []
    predictions = []
    truePosP = 0
    truePosN = 0
    falsePosN = 0
    falsePosP =0
    trueNegN = 0
    trueNegP = 0
    falseNegP = 0
    falseNegN = 0
    countCorrect = 0
    size = len(testData)
    
    for i in range(len(testData)):
        probN =  1
        probP =  1
        true_label = atts[-1].index(testData[i][-1]) 
        for k in range(len(testData[i]) -1): #ignore class label
             # calculate nominal att posterior probability
            if (atts[k][0] != "numeric"):
                idx = atts[k].index(testData[i][k])
                probN = probN *likelihoods[0][k][idx]
                probP = probP *likelihoods[1][k][idx]
            # calculate numeric att posterior probability
            else:
                x = int(testData[i][k])
                # do negative posterior - numeric - gaussian DF
                meanN = likelihoods[0][k][0]
                stdN = (likelihoods[0][k][1])# neg sd
                term1= 1/(math.sqrt(2*math.pi*stdN))
                exponent = math.exp(-(math.pow(x-meanN,2)/(2*math.pow(stdN,2))))
                g_prob_neg = ( term1 )* (exponent)
                #do positive posterior - numeric - gaussian DF
                meanP = likelihoods[1][k][0]
                stdP = (likelihoods[1][k][1])# neg sd
                term1= 1/(math.sqrt(2*math.pi*stdP))
                exponent = math.exp(-(math.pow(x-meanP,2)/(2*math.pow(stdP,2))))
                g_prob_pos =  ( term1 )* (exponent) 
                # add to each total the new product
                probN = probN *(g_prob_neg)
                probP = probP *(g_prob_pos)
        #multiply by the priors for each class
        probP = probP *( (classPriors[1]))
        probN = probN *  ((classPriors[0]))
        if probP > probN:
            pred_label = 1
            predictions.append(1)
        elif probP < probN:
            pred_label = 0
            predictions.append(0) 
        if true_label == pred_label:
            countCorrect += 1
            if true_label ==1:
                truePosP += 1
                trueNegN += 1
            elif true_label == 0:
                truePosN += 1
                trueNegP += 1 
        elif true_label != pred_label:
            if (true_label ==1) and (pred_label == 0):
                falsePosN +=1
                falseNegP += 1
            elif (true_label == 0) and (pred_label ==1):
                falseNegN +=1
                falsePosP +=1 
	# calculate evaluation metrics
    precisionN = truePosN/(truePosN +falsePosN)
    recallN = truePosN/ (truePosN + falseNegN) 
    precisionP =  truePosP/(truePosP + falsePosP)
    recallP= truePosP/ (truePosP + falseNegP)
    f1_measureN = (2*recallN*precisionN)/(recallN+precisionN)
    f1_measureP = (2*recallP*precisionP)/(recallP+precisionP)
    macro_precision = (precisionN + precisionP)/ 2
    macro_recall = (recallN + recallP)/ 2
    micro_precision = (truePosP + truePosN)/ (truePosP + truePosN + falsePosN + falsePosP)
    micro_recall = (truePosP + truePosN)/ (truePosP + truePosN + falseNegN + falseNegP)
    macro_f1 = (f1_measureN + f1_measureP)/ 2
    micro_f1 = 2*(micro_precision*micro_recall)/ (micro_precision + micro_recall)
    
    # show values for each round
    print("Micro precision:")
    micro_precision = (round(micro_precision, 4)) * 100
    print(micro_precision)
    microP.append(micro_precision)
    
    print("Micro recall:")
    micro_recall = (round(micro_recall, 4)) * 100
    print(micro_recall)
    microR.append(micro_recall)
    
    print("F1 micro: ")
    micro_f1 = (round(micro_f1, 4)) * 100
    print(micro_f1)
    microF1.append(micro_f1)
    
    print("Macro precision:")
    macro_precision = (round(macro_precision, 4)) * 100
    print(macro_precision)
    macroP.append(macro_precision)
    
    print("Macro recall:")
    macro_recall = (round(macro_recall, 4)) * 100
    print(macro_recall)
    macroR.append(macro_recall)
    
    print("F1 macro: ")
    macro_f1 = (round(macro_f1, 4)) * 100
    print(macro_f1)
    macroF1.append(macro_f1)  
 
    # get accuracy for this round
    correct  = countCorrect / size
    correct = correct * 100 
    correct = (round(correct, 2))
    accuracies.append(correct)
       
# MAIN
from scipy.io import arff
import numpy as np
import math
from os import path
import argparse
import sys
filename = sys.argv[-1]
import os
cwd = os.getcwd()
path = os.path.join(cwd, filename)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", dest="filename")
	parser.add_argument('--discretize', action='store_true')
	args = parser.parse_args()
	print(args.discretize)
	file = open(filename)
	k = 10
    atts = [] # array holds nominal att values, or 'numeric' att label, & numeric means
    instances = [] # array holds all the samples/instances
    trainSet= [] # array holds each round of training data
    testSet= [] # holds each round of testing data
    #performance metrics
    accuracies = []
    macroP = []
    macroR = []
    microP = []
    microR = []
    macroF1 = []
    microF1 = []
    # used for entropy discretization
    allGains = [] 
    attsValues = []
    bestSplits =[]
    pos = 4 # att to discretize - age  
    # get instances & atts from the file data & attributes
    parseFile(); 
    fold = round(len(instances) / k) 
    nominalAttCounts =  []
     
    #These methods will pre-process the data:
    #1) Calculate numeric and nominal means
    #2) Replace missing values in the entire data set
    #3) Calculate entropy and information gain to find best splitting
    # values for attribute specified by pos parameter ie. entropy based discretization        
    # att 4 = educ-numeric 
    #4) Update the attribute value for the data set by replace with bin value
      
    # append the mean to the numeric att in atts array
    calculateNumericAttMeans();
    # make nominalAttCounts hold counters of how many times each value
    # shows up in the examples and the last value of each array for each att is the mode
    calculateNominalAttModes(); 
    # replace missing nominal values with mode 
    # and missing numeric values with mean for that att
    replaceMissing();
    classAttEntropy = calculateEntropyOfClassAtt(-1); #class att
    # array of possible values for att being discretized
    attsValues =getAttsValues(pos,0, 1, attsValues)

    # perform discretization for att pos = position
    if args.discretize == 1:
        entropyDiscretize(classAttEntropy, pos, attsValues, 1); # specify which att?
        updateAttribute(pos, attsValues, bestSplits);
    #print(bestSplits)

    # End of pre-processing of data'''

    # Training and testing Naive Bayes classifier for the data:

    # Perform K fold cross validation to divide data into Train Set and Test Sets
    # Training
        #1) calculate Prior
        #2) calculate Likelihoods
    # Testing    
        #3) calculate Posterior 
        #4) store and output performance results
    
    kFoldCV();
    total = 0;
 
    for i in range(k):
        classPriors = []
        likelihoods = [] # 3D array; P/N class -> att # - > att length
        print(" " )
        print("             TESTING ROUND        ")
        print(i)
        calculateNBProb(trainSet[i])
        print("**********************************************************************")
        classifyNB(testSet[i])
        print("Accuracy: ")
        print(accuracies[i])
        total += accuracies[i]
        print(" ")
        print("**********************************************************************")
        print(" ")

        # show avg accuracy
        a = np.sum(accuracies)
        print("AVERAGE ACCURACY:")
        print(a/k)
        print(" ") 

        # show avg macro precision 
        a = np.sum(microP)
        print("AVERAGE MICRO PRECISION:")
        a = a/k
        print(a) 
        print(" ")

        # show avg micro recall 
        a= np.sum(microR)
        print("AVERAGE MICRO RECALL:")
        a = a/k
        print(a) 
        print(" ")

        #show avg micro F1
        a= np.sum(microF1)
        print("AVERAGE MICRO F1:")
        a = a/k
        print(a)
        print(" ") 

        # show avg macro precision
        a= np.sum(macroP)
        print("AVERAGE MACRO PRECISION:")
        a = a/k
        print(a)
        print(" ") 

        # show avg macro recall
        a= np.sum(macroR)
        print("AVERAGE MACRO RECALL:")
        a = a/k
        print(a)
        print(" ") 
         
        # show avg macro F1
        a= np.sum(macroF1)
        print("AVERAGE MACRO F1:")
        a = a/k
        print(a) 

     