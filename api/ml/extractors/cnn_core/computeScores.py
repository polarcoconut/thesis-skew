#
#
# This file is to take the output of MultiR along with the labeled test data
# and figure out the fscore, precision, recall and accuracy. 
#
#
#

import uuid
from datetime import datetime

relations = ['per:origin', '/people/person/place_of_birth', '/people/person/place_lived', '/people/deceased_person/place_of_death', 'travel']



def computeScoresAndAnalyze(predictedLabels, trueLabels,
                            test_examples, test_sentences,
                            classifier,features):

    
    logfile_name = '/homes/gws/chrislin/relex_data/test_data/temp/log-%s' % datetime.now()
    logfile = open(logfile_name, 'w')

    precisionNumerator = 0.0
    precisionDenominator = 0.0
    recallNumerator = 0.0
    recallDenominator = 0.0

    precisions = []
    recalls = []

    weights = classifier.coef_[0]
    #print "Num weights"
    #print len(weights)
    #print weights
    intercept = classifier.intercept_

    numFeatures = len(features)
    
    for trueLabel, predictedLabel, example, sentence in zip(
            trueLabels, predictedLabels,test_examples, test_sentences):
        
        if predictedLabel == 1:
            precisionDenominator += 1
            if predictedLabel == trueLabel:
                precisionNumerator += 1
                precisions.append(1)
            else:
                precisions.append(0)
        else:
            precisions.append('X')

        if trueLabel == 1:
            recallDenominator += 1
            if predictedLabel == trueLabel:
                logfile.write("#############GOOD RECALL#################\n")
                probs = classifier.predict_proba(example)[0]
                logfile.write("%f, %f" % (probs[0], probs[1]))
                logfile.write("\n")
                logfile.write(sentence)
                logfile.write("\n")

                #print example
                #print example.nonzero()
                nonzero_features = example.nonzero()[1]
                for feature in nonzero_features:
                    #print "Feature"
                    #print feature
                    logfile.write(features[feature])
                    logfile.write("\n")
                    logfile.write("%f" % weights[feature])
                    logfile.write("\n")

                recalls.append(1)
                recallNumerator += 1
            else:
                logfile.write("#############BAD RECALL#################\n")
                probs = classifier.predict_proba(example)[0]
                logfile.write("%f, %f" % (probs[0], probs[1]))
                logfile.write("\n")
                logfile.write(sentence)
                logfile.write("\n")
                #print example
                #print example.nonzero()
                nonzero_features = example.nonzero()[1]
                for feature in nonzero_features:
                    #print "Feature"
                    #print feature
                    logfile.write(features[feature])
                    logfile.write("\n")
                    logfile.write("%f" % weights[feature])
                    logfile.write("\n")
                    
                recalls.append(0)
        else:
            recalls.append('X')

    precision = precisionNumerator / precisionDenominator
    recall = recallNumerator / recallDenominator

    if precision + recall == 0:
        return (0, 0, 0)
    f1 = 2 * (precision * recall) / (precision + recall)

    #print "Intercept"
    #print intercept

    #print "# of positive instances"
    #print recallDenominator

    return (precision, recall, f1)
    
def computeScores(predictedLabels, trueLabels):
    precisionNumerator = 0.0
    precisionDenominator = 0.0
    recallNumerator = 0.0
    recallDenominator = 0.0

    precisions = []
    recalls = []
    for trueLabel, predictedLabel in zip(trueLabels, predictedLabels):
        if predictedLabel == 1:
            precisionDenominator += 1
            if predictedLabel == trueLabel:
                precisionNumerator += 1
                precisions.append(1)
            else:
                precisions.append(0)
        else:
            precisions.append('X')

        if trueLabel == 1:
            recallDenominator += 1
            if predictedLabel == trueLabel:
                recalls.append(1)
                recallNumerator += 1
            else:
                recalls.append(0)
        else:
            recalls.append('X')

    if precisionDenominator > 0:
        precision = precisionNumerator / precisionDenominator
    else:
        precision = 1.0

    if recallDenominator > 0:
        recall = recallNumerator / recallDenominator
    else:
        recall = 1.0

    if precision + recall == 0:
        return (0, 0, 0)
    f1 = 2 * (precision * recall) / (precision + recall)

    return (precision, recall, f1)

#testfile is the gold data file
#outputfile is what got output by the extractor
def computeScoresFromFile(outputfile, testfile, relInd):
    relation = relations[relInd]
    trueLabels = []



        
    for row in testfile:
        parts = row.split('\t')    
        y_gold_row = parts[7]

        if ','  in y_gold_row:
            y_gold_row = y_gold_row.split(',')
            y_gold_row[relInd] = y_gold_row[relInd].strip().strip('\'').strip('[').strip(']').strip('u\'')
            if 'neg' in y_gold_row[relInd]:
                trueLabels.append(0)
            else:
                trueLabels.append(1)
        else:
            if y_gold_row != relations[relInd]:
                trueLabels.append(0)
            else:
                trueLabels.append(1)
            
    predictedLabels = []
    for row in outputfile:
        row = row.split('\t')
        predictedRelation = row[3]
        #print predictedRelation
        #print relation
        #print predictedRelation == relation
        if predictedRelation == relation:
            predictedLabels.append(1)
        else:
            predictedLabels.append(0)

    return computeScores(predictedLabels, trueLabels)

