from parse import parse_test_data, parse_training_data
from cnn_core.train import train_cnn
from cnn_core.test import test_cnn
from cnn_core.computeScores import computeScores

def run_cnn(training_data_file_name, testing_data_file_name,
                            outputfile_name, cv=False):

    relations = ['per:origin', '/people/person/place_of_birth',
                 '/people/person/place_lived',
                 '/people/deceased_person/place_of_death', 'travel', 'NA']

    
    training_labels, training_examples,training_sentences = parse_training_data(
        training_data_file_name, 4, [])

    
    if cv:
        test_labels, test_examples, test_sentences = parse_training_data(
            testing_data_file_name, 4, [])
    else:
        test_labels, test_examples, test_sentences = parse_test_data(
            testing_data_file_name, [], 4)
        
    print "Training the CNN..."
    model_dir = train_cnn(training_sentences, training_labels)

    
    predicted_labels =  test_cnn(test_sentences, test_labels, model_dir)

    return (model_dir, computeScores(predicted_labels, test_labels))
