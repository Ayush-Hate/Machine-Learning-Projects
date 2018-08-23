# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:53:32 2018

This is an implementation of SIMPLE CART (Classification and Regression (binary) Trees) DECISION TREE CLASSIFIER which will 
use GINI INDEX as the SPLIT DETERMINOR,
take as input a training dataset,
train a model, 
display the decision tree generated,
predict a class y for an input x,
determine accuracy where accuracy is ratio of correct to wrong predictions
WORKS FOR 2 CLASSES ONLY (BINARY TREE IS GENERATED)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleDecisionTree:

    root = None    
    def __init__(self):
        self.root = dict()
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
    def getBestSplit(self, dataset):
        classValues = list(set(row[-1] for row in dataset)) #assuming class values are the last column of a dataset
        bestIndex, bestValue, bestGiniScore, bestGroups = 999,999,999,None
        for i in range(len(dataset[0])-1):
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
    
    def createIntermediateNodes(self, node, maxDepth, minSize, currentDepth):
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
            node['left'] = self.getBestSplit(left)
            self.createIntermediateNodes(node['left'],maxDepth,minSize,currentDepth+1)
        #Recursively split Right Child
        if len(right) <= minSize:
            node['right'] = self.createTerminalNode(right)
        else:
            node['right'] = self.getBestSplit(right)
            self.createIntermediateNodes(node['right'],maxDepth,minSize,currentDepth+1)
    
    def createTree(self,train,maxDepth,minSize):
        root = self.getBestSplit(train)
        self.createIntermediateNodes(root,maxDepth,minSize,1)
        self.root = root
        return root
    
    def printTree(self,node,depth = 0):
        if isinstance(node,dict):
            print('%s[X%d <- %.3f]' % ((depth*'\t', (node['index']+1), node['value'])))
            self.printTree(node['left'], depth+1)
            self.printTree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*'\t', node)))
    
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
            pred = self.predict(self.root,row)
            predicted.append(pred)
        positive = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                positive += 1
        accuracy = float(positive)/float(len(actual))*100.0
        print("\nAccuracy of model is: ",accuracy,"% ")
        return accuracy
    
#USEFUL FUNCTION TO CONVERT CSV STRINGS TO NUMBERS
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def testADataset(dataset,unknownRow):
    """
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        _column_to_float(dataset, i)
    """    
    SDT = SimpleDecisionTree()
    maxDepth = 5
    minSize = 10
    SDT.createTree(dataset,maxDepth,minSize)
    SDT.printTree(SDT.root)
    #print(dataset.head(10))
    SDT.performance(dataset)
    pred = SDT.predict(SDT.root,unknownRow)
    print(unknownRow," predicts -> class ",pred)    

#TEST 1 ON A HARDCODED 2 ATTRIBUTE DATASET
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]

#testADataset(dataset,[10,3,0])
print("__________________________________________________________________________\n")

#TEST ON BANKNOTES DATASET
data = np.loadtxt(open("banknote.csv" ,"rb"), delimiter=",", skiprows=1)
print(data.shape)
testADataset(data,[0,0,0,0])
print("__________________________________________________________________________\n")
#TEST ON PIMA DIABETES DATASET
data = np.loadtxt(open("indiansDiabetes.csv","rb"), delimiter=",")
print(data.shape)
testADataset(data,[1,85,66,29,0,26.6,0.351,31,0])
print("__________________________________________________________________________\n")

