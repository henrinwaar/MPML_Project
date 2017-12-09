# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:55:13 2017

@author: Max
"""

# Logistic Regression on Diabetes Dataset
from random import seed;
from random import randrange;
from math import exp;
import math as m;
import numpy as np;
import sklearn as sk;

###_____________________________________________________________________________________
###   Pre processing
###_____________________________________________________________________________________


# Load a CSV file
def load_csv(filename):
    with open(filename) as file:
        dataset = [];
        for line in file:
            # The rstrip method gets rid of the "\n" at the end of each line
            dataset.append(line.rstrip().split(" "));
    return dataset;

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip());

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list();
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset];
		value_min = min(col_values);
		value_max = max(col_values);
		minmax.append([value_min, value_max]);
	return minmax;

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
            
            
            

###_____________________________________________________________________________________
###  Modelling         
###_____________________________________________________________________________________


# Sigma function
def sigma(z):
    return 1/(1+exp(-z));
          
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list();
	dataset_copy = list(dataset);
	fold_size = int(len(dataset) / n_folds);
	for i in range(n_folds):
		fold = list();
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy));
			fold.append(dataset_copy.pop(index));
		dataset_split.append(fold);
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0;
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1;
	return correct / float(len(actual)) * 100.0;

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds);
	scores = list();
	for fold in folds:
		train_set = list(folds);
		train_set.remove(fold);
		train_set = sum(train_set, []);
		test_set = list();
		for row in fold:
			row_copy = list(row);
			test_set.append(row_copy);
			row_copy[-1] = None;
		predicted = algorithm(train_set, test_set, *args);
		actual = [row[-1] for row in fold];
		accuracy = accuracy_metric(actual, predicted);
		scores.append(accuracy);
	return scores;

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0];
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i];
	return sigma(yhat);

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_iter):
	coef = [0.0 for i in range(len(train[0]))];
	for index in range(n_iter):
		for row in train:
			yhat = predict(row, coef);
			error = row[-1] - yhat;
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat);
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i];
	return coef;

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_iter):
	predictions = list();
	coef = coefficients_sgd(train, l_rate, n_iter);
	for row in test:
		yhat = predict(row, coef);
		yhat = round(yhat);
		predictions.append(yhat);
	return(predictions);
    
def cost_function(data , n_iter, n_features, reg_lambda):
    coef = coefficients_sgd(data, 5, n_iter);
    cost = 0;
    for index in range(0, n_iter):
        y_hat = predict(data[index], coef);
        biais = 0;
        for index2 in range(0, n_features):
            biais += coef[index2] * coef[index2];
        biais = reg_lambda / (2 * n_iter) * biais;
        cost += (data[index][index2] * m.log10(y_hat)) + (1 - data[index][index2]) * (1 - m.log10(y_hat)) 
    cost = (-1/n_iter) * cost + biais;
    return cost;
###_______________________________________________________________________________________________
### Test the logistic regression algorithm on the spam dataset
###_______________________________________________________________________________________________
    
    
    
seed(1);

# load and prepare data
filename = 'C:/Users/Max/Documents/UPM/Massive_Parallele_Machine_Learning/Project/spam.txt'
dataset = load_csv(filename);
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i);
    
# normalize
minmax = dataset_minmax(dataset);
normalize_dataset(dataset, minmax);

# evaluate algorithm
n_folds = 8;
l_rate = 5;
n_iter = 100;
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_iter);
print('Scores: %s' % scores);
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))));

cost = cost_function(dataset, 100, 58, 5)