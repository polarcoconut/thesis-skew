import os
import numpy as np

# express the test data in the feature space
# allow multiple positive relations in one sentence
# return: y_gold, sparse_matrix
# return: X_test, sparse_matrix
def parse_test_data(testFile, allFeatures, relId):
	lenFeatures = len(allFeatures)

	# generate class expression
	# relation = ['per:origin', '/people/person/place_of_birth', '/people/person/place_lived', '/people/deceased_person/place_of_death', 'travel']

	# testFeatures = {}
	num = getLen(testFile)

	y_gold = lil_matrix((num, 5), dtype=np.int8)
        y_gold_single = np.zeros(num, dtype=np.int8)
	X_test = lil_matrix((num, lenFeatures), dtype=np.int8)

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
			for i in range(5):
				y_gold_row[i] = y_gold_row[i].strip().strip('\'').strip('[').strip(']').strip('u\'')
				#if y_gold_row[i] == 'optional':
                                #y_gold[count, i] = -1
				if 'neg' in y_gold_row[i]:
					y_gold[count, i] = 0
                                else:
                                        y_gold[count, i] = 1
                        if 'neg' in y_gold_row[relId]:
                                y_gold_single[count] = 0
                        else:
                                y_gold_single[count] = 1
			# process the feature vector
			lenParts = len(parts)
			featureWalker = 12
			while featureWalker < lenParts:
				feature = parts[featureWalker]
				if feature in allFeatures:
					X_test[count, allFeatures[feature]] = 1
				featureWalker += 2

			count += 1

	#print "Test Data Ready"
        #print "Here's data about the skew"
        #skew = 0.0
        #for label in y_gold_single:
        #        if label == 1:
        #                skew += 1
        #skew /= len(y_gold_single)
        #print skew

	# lenTestFeatures = len(testFeatures)
	# print "%d features in the test data" % lenTestFeatures
	return y_gold_single, X_test, sentences

def getLen(inputFile):
    if os.stat(inputFile).st_size == 0:
	return 0
    with open(inputFile) as fIn:
	for i, l in enumerate(fIn):
	    pass
    return i + 1
