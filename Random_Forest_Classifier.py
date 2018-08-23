# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:44:51 2018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
import math

class SimpleRandomForest:

    root = None
    trees= None    
    def __init__(self):
        self.root = dict()
        self.trees=list()
    #function to evaluate a gini index for a split
    def giniIndex(self,groups,classes):
        totalInstances = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if(size == 0):
                continue
            score = 0.0
            for classValue in classes:
                score += (([row[-1] for row in group].count(classValue)/size)**2)
            gini += (1.0-score) * (size/float(totalInstances))
        return gini
    
    #function to split dataset
    def splitData(self, indexOfSplitAttribute, thresholdValue, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[indexOfSplitAttribute]<thresholdValue:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    #function to evaluate the gini index of all splits to select the best split attribute
    def getBestSplit(self, dataset,numFeatures):
        classValues = list(set(row[-1] for row in dataset)) #assuming class values are the last column of a dataset
        bestIndex, bestValue, bestGiniScore, bestGroups = 999,999,999,None
        features = list()
        while len(features) < numFeatures:
            index = rnd.randint(0,len(dataset[0])-1)
            if index not in features and index != len(dataset[0])-1:
                features.append(index)
        for i in features:
            for row in dataset:
                groups = self.splitData(i,row[i],dataset)
                gini = self.giniIndex(groups,classValues)
                if gini<bestGiniScore:
                    bestIndex,bestValue,bestGiniScore,bestGroups = i,row[i],gini,groups
        return {'index':bestIndex, 'value':bestValue, 'groups':bestGroups} #returns a dictionary
    
    #following functions are for the creation and manipulation of the Decision TREE
    
    def createTerminalNode(self, group):
        classes = [row[-1] for row in group]
        return max(set(classes), key=classes.count) #returns the most common class label for a terminal node
    
    def createIntermediateNodes(self, node, maxDepth, minSize, currentDepth,numFeatures):
        left, right = node['groups']
        del(node['groups'])
        #Check if no node can be split
        if not left or not right:
            node['left'] = node['right'] = self.createTerminalNode(left+right)
            return
        #check if user specified max depth has been reached
        if currentDepth>=maxDepth:
            node['left'], node['right'] = self.createTerminalNode(left), self.createTerminalNode(right)
            return
        #Recursively split Left Child
        if len(left) <= minSize:
            node['left'] = self.createTerminalNode(left)
        else:
            node['left'] = self.getBestSplit(left,numFeatures)
            self.createIntermediateNodes(node['left'],maxDepth,minSize,currentDepth+1,numFeatures)
        #Recursively split Right Child
        if len(right) <= minSize:
            node['right'] = self.createTerminalNode(right)
        else:
            node['right'] = self.getBestSplit(right,numFeatures)
            self.createIntermediateNodes(node['right'],maxDepth,minSize,currentDepth+1,numFeatures)
    
    def createTree(self,train,maxDepth,minSize,numFeatures):
        root = self.getBestSplit(train,numFeatures)
        self.createIntermediateNodes(root,maxDepth,minSize,1,numFeatures)
        self.root = root
        return root
        
    #prediction is basically traversing a decision tree given an input row and finding the class label        
    def predict(self,node, row):
        if(row[node['index']] < node['value']):
            if isinstance(node['left'],dict):
                return self.predict(node['left'],row)
            else:
                return node['left']
        else:
            if isinstance(node['right'],dict):
                return self.predict(node['right'],row)
            else:
                return node['right']
    
    def performance(self, dataset):
        actual = [row[-1] for row in dataset]
        predicted = list()
        for row in dataset:
            mPred,aPred = self.baggingPredictionUsingAllTrees(self.trees,row)
            predicted.append(mPred)
        positive = 0
        negative = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                positive += 1
            else: 
                negative += 1
        accuracy = float(positive)/float(len(actual))*100.0
        print("\nPositive: ",positive,"\nNegative: ",negative,"Accuracy of model is: ",accuracy,"% \n")
        return accuracy
    
    #function to create a random sample with replacement from dataset
    def randSample(self, dataset, ratio):
        sample = list()
        numElements = round(len(dataset)*ratio)
        while len(sample) < numElements:
            i = rnd.randint(0,len(dataset)-1)
            sample.append(dataset[i])
        return sample
    
    def baggingPredictionUsingAllTrees(self,trees,row):
        predictions = [self.predict(tree,row) for tree in trees]
        maxPred = max(set(predictions),key=predictions.count)
        avgPred = float(sum(predictions)/len(predictions))
        return (maxPred, avgPred) #max is the class label with most votes avg is an average label

    def randomForest(self,train,test,maxDepth,minSize,numElements,numTrees,numFeatures):
        trees=list()
        maxPredictions=list()
        avgPredictions=list()
        for i in range(numTrees):
            sample = self.randSample(train,numElements)
            tree = self.createTree(sample,maxDepth,minSize,numFeatures)
            trees.append(tree)
        self.trees=trees
        for row in test:
            maxP,avgP = self.baggingPredictionUsingAllTrees(trees,row)
            maxPredictions.append(maxP)
            avgPredictions.append(avgP)
        return maxPredictions,avgPredictions
    
def testExample(dataset,unknownRow,testSet):
    """
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        _column_to_float(dataset, i)
    """    
    SRF = SimpleRandomForest()
    maxDepth = 15
    minSize = 5
    numFeatures = int(math.sqrt(len(dataset[0])-1))
    SRF.createTree(dataset,maxDepth,minSize,numFeatures)
    sampleRatio=0.50
    accuracyList=list()
    numTreeList=list()
    maxAccuracy = 0
    bestNumTree = 0
    for numTrees in range(100,1001,100): #optimal 65-85 and "10"
 
#        maxP,avgP = SRF.randomForest(dataset,testSet,maxDepth,minSize,sampleRatio,numTrees,numFeatures)
        
        maxP,avgP = SRF.randomForest(dataset,unknownRow,maxDepth,minSize,sampleRatio,numTrees,numFeatures)
        
        maxPred = max(set(maxP),key = maxP.count)
        avgPred = float(sum(avgP)/len(avgP))
        print(unknownRow," predicts -> class ",maxPred," at maximum\nAnd predicts -> class ",avgPred," on average of all trees\n0->R 1->M\n" )
        accuracy=SRF.performance(dataset)
        accuracyList.append(accuracy)
        numTreeList.append(numTrees)
        if accuracy>maxAccuracy:
            maxAccuracy = accuracy
            bestNumTree = numTrees
    #plot of accuracy with respect to various number of trees
    plt.plot(numTreeList,accuracyList,'b-')
    #from this plot we can find the maximum numTree value for maxmimzing accuracy of model
    print("\nOptimal number of trees is ",bestNumTree," yielding accuracy of ",maxAccuracy,"%\n")
#TEST ON SONAR DATASET
data = np.loadtxt(open("sonar.csv" ,"rb"), delimiter=",")
print(data.shape)
#testExample(data,[[0.031,0.0221,0.0433,0.0191,0.0964,0.1827,0.1106,0.1702,0.2804,0.4432,0.5222,0.5611,0.5379,0.4048,0.2245,0.1784,0.2297,0.272,0.5209,0.6898,0.8202,0.878,0.76,0.7616,0.7152,0.7288,0.8686,0.9509,0.8348,0.573,0.4363,0.4289,0.424,0.3156,0.1287,0.1477,0.2062,0.24,0.5173,0.5168,0.1491,0.2407,0.3415,0.4494,0.4624,0.2001,0.0775,0.1232,0.0783,0.0089,0.0249,0.0204,0.0059,0.0053,0.0079,0.0037,0.0015,0.0056,0.0067,0.0054,0]],list(data[0:100,]))
print("__________________________________________________________________________\n")
