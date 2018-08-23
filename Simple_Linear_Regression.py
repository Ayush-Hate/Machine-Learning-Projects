# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:46:15 2018

This is an implementation of SIMPLE LINEAR REGRESSION which will 
take as input a training dataset,
train a model by calculating coefficients, 
display the regression line plot,
predict a y for an input x,
determine accuracy where accuracy is ratio of correct to wrong predictions
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SimpleLinearRegression:
    
    dataset = np.empty
    x = np.empty
    y = np.empty
    slope = 0.0
    intercept = 0.0  
    
    def __init__(self,dataset):
        self.dataset = np.array(dataset)
        self.x = np.array(dataset[:,0], dtype=np.float64)
        self.y = np.array(dataset[:,1], dtype=np.float64)
    
    def plotDataset(self):
        plt.plot(self.dataset[:,0],self.dataset[:,1],'ro')
        if(self.slope != 0 or self.intercept != 0):
            axes = plt.gca()
            x_values = np.array(axes.get_xlim())
            y_values = self.intercept + self.slope*x_values
            plt.plot(x_values,y_values,'--')
            plt.show()
        
    def train(self):
        meanX = np.mean(self.x)
        meanY = np.mean(self.y)
        sum1 = 0.0
        sum2 = 0.0
        for i in range(self.x.shape[0]):
            sum1 += (self.x[i]-meanX)*(self.y[i]-meanY)
            sum2 += (self.x[i]-meanX)**2
        self.slope = sum1/sum2
        self.intercept = meanY - (self.slope*meanX)
        print("\nLinear Model Trained with \nSlope = ",self.slope,"\nIntercept = ",self.intercept)
        self.plotDataset()
    
    def predict(self,test_X):
        Y = self.intercept + self.slope*test_X
        print("For given X = ",test_X," the Linear Model estimates Y = ",Y)
        plt.plot(test_X,Y,'bo',markersize=10)
        self.plotDataset()
        return Y
    
    def performance(self,trainingDataset):
        testX = np.array(trainingDataset[:,0])
        testY = np.array(trainingDataset[:,1])
        total = testX.shape[0]
        positive = 0
        for i in range(total):
            #predY = self.predict(testX[i])
            predY = self.intercept + self.slope*testX[i]
            if abs(predY-testY[i]) <=0.005:
                positive += 1
        accuracy = float(positive)/float(total)*100.0
        print("\nAccuracy of model is: ",accuracy,"% ")
        
        
def testExample(dataset):
    SLR = SimpleLinearRegression(dataset)
    SLR.plotDataset()
    SLR.train()
    SLR.predict(0)
    SLR.performance(dataset)

#FOLLOWING ARE TEST CASES USING LINEAR REGRESSION TO PLOT A BEST FIT LINE
dataset = np.array([[0,0],[1,1],[2,2],[3,3]])
testExample(dataset)

from sklearn import datasets
diabetes = datasets.load_diabetes()
data = diabetes.data[:,0:10:9]
testExample(data)

data = np.loadtxt(open("AnimalsBodyBrain.csv" ,"rb"), delimiter=",", skiprows=1,usecols = (1,2))
print(data.shape)
testExample(data)

data = np.loadtxt(open("AmesHousing.csv" ,"rb"), delimiter=",", skiprows=1,usecols = (5,81))
print(data.shape)
testExample(data)

