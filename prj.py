# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:37:40 2017

@author: Max
"""

import numpy as np;
import math as math;
from sklearn.preprocessing import normalize;

## Import the data as a list of list, each element of the list is a record which is a list of the value
## of the 56 features and a 1 if it's a spam or a 0 if it's not

with open("C:/Users/Max/Documents/UPM/Massive_Parallele_Machine_Learning/Project/spam.txt") as file:
    dataInLine = [];
    for line in file:
        # The rstrip method gets rid of the "\n" at the end of each line
        dataInLine.append(line.rstrip().split(" "));
        
## Cast all the elements to float and normalizing the data

dataInLine = normalize(dataInLine);     
for element1 in dataInLine:
    for element2 in element1:
        element2 = float(element2);
        

## Split the data set into two parts: train and test

split_index = int(0.75*len(dataInLine));
train = dataInLine[:split_index];
test = dataInLine[split_index:];

##______________________________________________________________________________________________________

def sigma(z):
    return 1/(1+math.exp(-z));
        
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0];
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i];
	return sigma(-yhat);
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))];
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef);
			error = row[-1] - yhat;
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat);
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i];
	return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list();
	coef = coefficients_sgd(train, l_rate, n_epoch);
	for row in test:
		yhat = predict(row, coef);
		yhat = round(yhat);
		predictions.append(yhat);
	return(predictions);
    
reg = logistic_regression(train, test, 0.4, 10000);