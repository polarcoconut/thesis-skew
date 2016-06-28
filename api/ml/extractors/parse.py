import os
import numpy as np

# relation-wise training data collection
def parse_training_data(trainingFile, relInd, allFeatures):

        y = []
        sentences = []
        with open(trainingFile) as f:
                for row in f:
                        row = row.split('\t')
                        print row
                        sentence = row[0]
                        label = int(row[1])
                        sentences.append(sentence)
                        y.append(label)
                return y, [], sentences
        

# express the test data in the feature space
# allow multiple positive relations in one sentence
# return: y_gold, sparse_matrix
# return: X_test, sparse_matrix
def parse_test_data(testFile, allFeatures, relId):
	lenFeatures = len(allFeatures)

	num = getLen(testFile)

        y_gold_single = []
        sentences = []
	count = 0
	with open(testFile) as f_test:
		for row in f_test:
			parts = row.split('\t')
			# process the gold annotation item
			y_gold_row = parts[7]
                        sentence = parts[11]
                        sentences.append(sentence)
			y_gold_row = y_gold_row.split(',')

                        if 'neg' in y_gold_row[relId]:
                                y_gold_single.append(0)
                        else:
                                y_gold_single.append(1)
			# process the feature vector
			count += 1
	return y_gold_single, [], sentences

def getLen(inputFile):
    if os.stat(inputFile).st_size == 0:
	return 0
    with open(inputFile) as fIn:
	for i, l in enumerate(fIn):
	    pass
    return i + 1
