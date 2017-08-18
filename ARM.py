'''
Association Rule Mining
Data Mining
'''
import numpy as np 
import math
import collections
import time
import argparse
import sys
from os import path
import os

MIN_SUPPORT = 0.30
MIN_CONF = 0.65

# Get products data
def getProducts():
    # representing items sets
    for p in products:
        p = p.split(",")
        productsList.append(p[0])
    return productsList
    
# Return frozen set 
def getItemSet():
    D = []
    with open(small_basket) as smallbasket:
        for line in smallbasket:
            arr = []
            line = line.split(",") 
            for i in range(1, len(line)): 
                if (line[i] != ' 0') and (line[i] != ' 0\n'):
                    val = int(line[i])       
                    item = productsList[i-1]
                    idx = productsList.index(item)
                    items.add(item) 
                    arr.append(idx)
            D.append(arr)
    return D
    
# Get list of k=1 items
def getCandidates():
    for c in c1:
        i = frozenset({c})
        candidatesSet.append(i)
    return candidatesSet

# Create candidate item sets size k = 1  - C1
def createC1(data):
    c1 =  set()
    for t in data:
        for item in t : 
            if not item in c1:
                c1.add(item)
    return c1    

# Check min Support of itemsets 
def scanData(d, candidates, MIN_SUPPORT, minSupportData):
    L = {}
    L ={ frozenset({0}):0}
    freqItems = []
    dataSize = len(d)
    # check if itemset is part of this transaction 
    # create L1 hash map with freqset and item support 
    for tid in d:
        s = set(tid)
        for c in candidates:
            if c.issubset(s):
                L.setdefault(c, 0) #sets value 0 if item not in dict
                L[c] += 1 
    # keep itemsets that meet minSupport
    for i in L:
        support = L[i] /dataSize
        if support  >= MIN_SUPPORT: 
            if (bool(minSupportData)):
                minSupportData[i] = L[i]
            else:
                minSupportData = {i:L[i]}
            freqItems.append(i)
    allsup.append(minSupportData)
    return freqItems, minSupportData

# Generate frequent candidate itemsets of k + 1
def aprioriJoin(freq, k):
    joinSets = []    
    dataSize = len(freq)
    for i in range(dataSize):
        for j in range( i+1, dataSize):
            L1 = list(freq[i])[:k-2]
            L2 = list(freq[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2: 
                joinSets.append(freq[i]  | freq[j])
    return joinSets

# Generate k >2 size itemsets
def apriori(data, MIN_SUPPORT, minSupportData):
    # genereate candidates itemsets
    k = 2
    while(len(allfreq[k-2]) >0):
        Ck = aprioriJoin(allfreq[k-2],k)
        Lk, minSupportData = scanData(data, Ck, MIN_SUPPORT, minSupportData)
        allfreq.append(Lk)
        k += 1  
    allsup.append(minSupportData) 
    return allfreq
     
# Get association rules    
def getRules(allfreq, allsup, MIN_CONF):
     
    for k in range(1, len(allfreq)): # represents k items 
        for freqset in allfreq[k]:
            i = [frozenset([item]) for item in freqset ]   
            if ( k > 1 ):
                calcRules(freqset, i, allsup, aRules, MIN_CONF)
            else:
                calcConfidence(freqset, i, allsup, aRules, MIN_CONF)
    return aRules 
	
# Calculate rules
def calcRules( freqSet, i, allsup,  aRules, MIN_CONF ):
    n = len(i[0])
    if len(freqSet) > (n+1):
        j = aprioriJoin(i, n+1)
        j = calcConfidence(freqSet, j, allsup, aRules, MIN_CONF)
        if len(j) > 1: 
            calcRules( freqSet, j, allsup,  aRules, MIN_CONF)
    
# Calcualte and check Min confidence
def calcConfidence( freqSet, i, allsup,  aRules, MIN_CONF):
    p = []
    for c in i: 
        conf = allsup[0][freqSet] / allsup[0][freqSet - c]
        if conf >= MIN_CONF: 
            conf = round(conf, 4)
            aRules.append((freqSet -c , c, conf))
            p.append(c)
    return p 

# Translate the association rules into actual product names
def getItemNames(aRules):
    aRulesNames  = []
    for rule in aRules: 
        ruleSet = [] 
        for i in range(len(rule) - 2):
            if(len(rule[i]) > 1):
                arr = []
                for k in rule[i]:
                    ruleName = productsList[k]
                    arr.append(ruleName) 
                ruleSet.append(arr)
            else:
                (element,) = rule[i] 
                ruleName = productsList[element] 
                ruleSet.append(ruleName)
        # append confidence term
        ruleSet.append(rule[len(rule)-2])        
        ruleSet.append(rule[len(rule)-1])
        aRuleNames.append(ruleSet)
                
    return aRuleNames

# Calcualte Lift - prune rules set to remove neg correlations rules
def calcLift(aRules):
    dataSize = len(data) 
    for i in range(len(aRules)): 
        u = aRules[i][0].union(aRules[i][1])
        v1 = (allsup[0][aRules[i][0]])/dataSize
        v2 = (allsup[0][aRules[i][1]])/dataSize
        u  = allsup[0][u]/dataSize
        lift = round( u / (v1 *v2), 4)
         # append lift value after conf value
        aRules[i] = aRules[i] + (lift,)

    # filter out negative correlation 
    for rule in aRules:
        if rule[-1] < 1: 
            idx = aRules.index(rule)
            aRules.pop(idx)
    return aRules
        
      
# MAIN*****************************************

products  = sys.argv[-1]
small_basket = sys.argv[-2]
 
cwd = os.getcwd()
path = os.path.join(cwd, products)
path = os.path.join(cwd, small_basket)
products = open(products) 
#smallbasket = open(small_basket) 

data = []
productsList = [] #product names strings
candidatesSet = [] #products ints
items = set() # items size 1 
minSupportData = {}
getProducts() # get products list
data = getItemSet() # get tids for each item  - numerical
data = list(map(set,data)) #map each transaction as a set
allfreq = [] # keep track of all freq sets
allsup = [] # keep track of all supports
aRules = [] # keep track of rules with indices
aRuleNames = [] #keep track of rules with string names

c1 = createC1(data) # generate k = 1 items
c1 = list(c1)
candidatesSet = []
# genereate full list of k=1 size sets of the items (0-99)
candidatesSet = getCandidates()
# generate first itemset k = 1 with min Support calculation

start = time.time()
L1, minSupportData = scanData(data,candidatesSet ,MIN_SUPPORT, minSupportData)
allfreq.append(L1)
# genereate k > 1 itemsets 
allfreq = apriori(data, MIN_SUPPORT, minSupportData)
#generate association rules from the frequent itemsets 
getRules( allfreq , allsup,  MIN_CONF)
# prune rules set to remove negative correlations rules <1 
aRules = calcLift(aRules)
# get names of products for the rules
aRuleNames = getItemNames(aRules)

print("*******************RULES******************")
print(" ")
for r in aRuleNames:
    print(r)
print(" " )
for i in range(len(allfreq)):
    print("Size k: " , i+1, "; Frequent sets : " ,len(allfreq[i]))
    print("..................................")
stop = time.time()

print("Total Rules : " , len(aRuleNames))
print(" " )
print("--- %s seconds  ---" % (stop - start ))
  