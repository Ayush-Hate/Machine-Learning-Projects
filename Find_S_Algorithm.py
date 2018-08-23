# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:12:14 2018

@author: atulh
"""

def loadCSV(filename,skipHeader):
    with open(filename,'r') as file:
        f = list()
        for line in file:
            if skipHeader==1:
                skipHeader-=1
                continue
            line=line.rstrip()
            word = line.split(',')
            f.append(word)
        return len(f[0]),f

def find_S_train(trainData,col):
    h=list()
    for i in range(col-1):
        h.append('-')
    print("\nBy Find-S Algorithm:\nInitially: Hypothesis, h = ",h)
    initialFlag=1
    for row in trainData:
        index = 0
        if row[-1] == "No":
            print("h: Skip negative example!")
            continue
        if initialFlag == 1:
            initialFlag=0
            for i in range(col-1):
                h[i]=row[i]
        for attribute,hypothesisAttribute in zip(row,h):
            if attribute == hypothesisAttribute:
                index+=1
                continue
            h[index]="?"
            index+=1
        print("h: ",h)
    return h

def find_S_test(testData,col,h):
    predictedOutput = list()
    expectedOutput = list()
    for row in testData:
        expectedOutput.append(row[-1])
        yesFlag=1
        for i in range(col-1):
            if h[i] == "?" or h[i] == row[i]:
                continue
            else:
                yesFlag=0
                predictedOutput.append("No")
                break
        if yesFlag == 1:
            predictedOutput.append("Yes")
    positiveCount = 0
    print("\nTesting the dataset with hypothesis h = ",h,":\nExpected Output\tPredicted Output\n_______________________________________")
    for expectedOP, predictedOP in zip(expectedOutput,predictedOutput):
        if expectedOP == predictedOP:
            positiveCount+=1
        print(expectedOP,"\t\t",predictedOP)
    print("\nAccuracy of h is: ",((float(positiveCount)/float(len(testData)))*100),"%")

def main():
    col,file=loadCSV("lab1.csv",1)
    print("Find-S Algorithm\n\nCSV Data of ",col," columns loaded as list:\n")
    for row in file:
        print(row)
    hypothesis = find_S_train(file,col)
    find_S_test(file,col,hypothesis)
    
main()