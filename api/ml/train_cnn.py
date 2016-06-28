from constructTrainingData import constructTrainingData
from computeScores import computeScores
from extractors.cnn import run_cnn

################
# CONFIGURATION
relations = ['nationality', 'born', 'lived', 'died', 'travel']
amount_of_data = [1898, 496, 3897, 1493, 1992]

#Extractor Choices: 'multir', 'lr', 'mimlre', 'cnn'
#extractor = 'multir' 
#extractor = 'lr'
extractor = 'cnn'
#extractor = 'mimlre'

len_positive_data = 400
negative_to_positive_ratio = 1.0
base_negative_to_positive_ratio = 1.0
#Where to put multir results
outputfile_name = 'extremestextraction_results'

testfile_name = 'test_strict_new_feature'

################
            

def trainCNN(positive_examples, negative_examples):

        
    training_data_file_name = constructTrainingData(positive_examples,
                                                    negative_examples)
    
    (model_dir, results) = run_cnn(
        training_data_file_name,
        testfile_name,
        outputfile_name, False)

    precision, recall, f1 = results

    print "Results on the test file:"
    print "Precision: %f" % precision
    print "Recall: %f" % recall
    print "F1: %f" % f1

    return model_dir
