from constructTrainingData import constructTrainingData
from computeScores import computeScores
#from extractors.cnn import run_cnn
from extractors.parse import parse_test_data, parse_training_data
from extractors.cnn_core.train import train_cnn
from extractors.cnn_core.test import test_cnn
from extractors.cnn_core.computeScores import computeScores
from app import app

################
# CONFIGURATION
relations = ['nationality', 'born', 'lived', 'died', 'travel']
amount_of_data = [1898, 496, 3897, 1493, 1992]

testfile_name = 'test_strict_new_feature'

################
            

def trainCNN(positive_examples, negative_examples):

            
    model_file_name, vocabulary = train_cnn(
        positive_examples + negative_examples,
        [1 for e in positive_examples] + [0 for e in negative_examples])

    test_labels, test_examples, test_sentences = parse_test_data(
        testfile_name, [], 4)

    predicted_labels =  test_cnn(test_sentences, test_labels,
                                 model_file_name, vocabulary)


    precision, recall, f1 = computeScores(predicted_labels, test_labels)

    print "Results on the test file:"
    print "Precision: %f" % precision
    print "Recall: %f" % recall
    print "F1: %f" % f1

    
    return model_file_name, vocabulary
