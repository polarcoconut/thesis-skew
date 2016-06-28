from sklearn.linear_model import LogisticRegression
from computeScores import computeScores, computeScoresAndAnalyze
from collections import OrderedDict
import os
import numpy as np
from scipy.sparse import *
from parse import parse_test_data, parse_training_data

def run_logistic_regression(training_data_file_name, testing_data_file_name,
                            outputfile_name, relInd, cv=False):

    relations = ['per:origin', '/people/person/place_of_birth',
                 '/people/person/place_lived',
                 '/people/deceased_person/place_of_death', 'travel', 'NA']

    #training_data_file = open(training_data_file_name, 'r')
    #testing_data_file = open(testing_data_file_name, 'r')

    allFeatures, inverse_allFeatures = getFeatures(training_data_file_name, 0)

    training_labels, training_examples,training_sentences = parse_training_data(
        training_data_file_name, relInd, allFeatures)

    if cv:
        test_labels, test_examples, test_sentences = parse_training_data(
            testing_data_file_name, relInd, allFeatures)
    else:
        test_labels, test_examples, test_sentences = parse_test_data(
            testing_data_file_name, allFeatures, relInd)

    classifier = LogisticRegression(penalty='l2', C=1.0)
    classifier.fit(training_examples, training_labels)

    weights = classifier.coef_
    intercept = classifier.intercept_
    
    predicted_labels = classifier.predict(test_examples)


    if cv:
        return computeScoresAndAnalyze(predicted_labels, test_labels,
                                       test_examples, test_sentences,
                                       classifier,
                                       inverse_allFeatures)
    else:
        return computeScores(predicted_labels, test_labels)
    



# get every feature with its counts and do feature pruning
# pruningThres: features that occur less than pruningThres will be pruned
def getFeatures(stdFile, pruningThres):
	allFeatures = OrderedDict()
	featureCount = 0

	# load all the features
	with open(stdFile) as f:
		for row in f:
			parts = row.split('\t')
			lenParts = len(parts)
			featureWalker = 12
			while featureWalker < lenParts:
				feature = parts[featureWalker]
				if feature not in allFeatures:
					featureCount += 1
					allFeatures[feature] = 1
				else:
					allFeatures[feature] += 1
				featureWalker += 2

	# feature pruning
	if pruningThres > 0:
		for key in allFeatures:
			if allFeatures[key] < pruningThres:
				del allFeatures[key]

	# use the value as index to have O(1) time
	featureIndex = 0
        inverse_allFeatures = {}
	for key in allFeatures:
		allFeatures[key] = featureIndex
                inverse_allFeatures[featureIndex] = key
		featureIndex += 1

	#print len(allFeatures), "features loaded"
	return allFeatures, inverse_allFeatures

