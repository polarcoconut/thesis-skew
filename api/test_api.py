from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
import sys
import pickle
#from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores
from ml.extractors.cnn_core.parse import parse_angli_test_data, parse_tackbp_test_data
from util import write_model_to_file, getLatestCheckpoint, split_examples, retrain, test
from schema.job import Job
import numpy as np
import traceback
import time
import requests

test_parser = reqparse.RequestParser()
test_parser.add_argument('job_id', type=str, required=True)
test_parser.add_argument('test_sentence', type=str, required=True)
#test_parser.add_argument('positive_types', required=True,
#                            action='append')

cv_parser = reqparse.RequestParser()
cv_parser.add_argument('job_id', type=str, required=True)
cv_parser.add_argument('test_set', required=True)
cv_parser.add_argument('positive_types', required=True,
                            action='append')

class TestExtractorApi(Resource):
    def get(self):

        args = test_parser.parse_args()
        job_id = args['job_id']
        test_sentence = args['test_sentence']
        #positive_types = args['positive_types']

        print "Extracting event from sentence:"
        print test_sentence
        sys.stdout.flush()

        predicted_labels = test(job_id, [test_sentence], [0])

        print "predicted_labels"
        print predicted_labels
        sys.stdout.flush()
            
        return predicted_labels[0]

class CrossValidationExtractorApi(Resource):
    def get(self):

        args = cv_parser.parse_args()
        job_id = args['job_id']
        positive_types = args['positive_types']
        test_set = int(args['test_set'])
        
        print "Testing on  held-out set %d" % test_set
        sys.stdout.flush()

        return test_on_held_out_set(job_id, positive_types, test_set)



def test_on_held_out_set(job_id, positive_types, test_set):
    job = Job.objects.get(id = job_id)
                
    test_set = test_set.split('\t')
    if len(test_set) == 2:
        #Otherwise, if the test set contains urls
        (positive_testing_examples_url,
         negative_testing_examples_url) = test_set
        while True:
            try:
                r = requests.get(positive_testing_examples_url).content
                break
            except Exception:
                print "Exception while communicating with S3"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(60)
                continue
                    
        test_positive_examples = str(r).split('\n')

        while True:
            try:
                r = requests.get(negative_testing_examples_url).content
                break
            except Exception:
                print "Exception while communicating with S3"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(60)
                continue
                    
        test_negative_examples = str(r).split('\n')

        test_examples = test_positive_examples + test_negative_examples
        test_labels = ([1 for e in test_positive_examples] +
                       [0 for e in test_negative_examples])

    else:
        test_set = int(test_set[0])
        ####
        # Each of these options should return
        # test_examples, test_positive_examples
        # test_negative_examples and test_labels
        ####
        if test_set == -1:
            checkpoint = getLatestCheckpoint(job_id)
            (task_information, budget) = pickle.loads(job.task_information)        
            (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

            test_positive_examples, test_negative_examples = split_examples(
                task_ids[0:2],
                task_categories[0:2], positive_types)
            test_examples = test_positive_examples + test_negative_examples
            test_labels = ([1 for e in test_positive_examples] +
                           [0 for e in test_negative_examples])
        elif test_set >= 0 and test_set <= 4:
            relations = ['nationality', 'born', 'lived', 'died', 'travel']
            amount_of_data = [1898, 496, 3897, 1493, 1992]
            testfile_name = 'data/test_data/test_strict_new_feature'
            (test_labels, test_features, test_examples,
             test_positive_examples,
             test_negative_examples) = parse_angli_test_data(
                 testfile_name, [], test_set)
        elif test_set >= 5 and test_set <= 9:
            relations = ['transfermoney', 'broadcast', 'attack', 'contact',
                         'transferownership']
            testfile_name = 'data/test_data/testEvents'
            relation = relations[test_set-5]
            (test_positive_examples, 
             test_negative_examples) = parse_tackbp_test_data(testfile_name, 
                                                              relation)
            test_examples = test_positive_examples + test_negative_examples
            test_labels = ([1 for e in test_positive_examples] +
                           [0 for e in test_negative_examples])
    #else:
        #This is to test using generated examples from the crowd
    #    test_positive_examples = []
    #    test_negative_examples = []
    #    pos_testfile_name = 'data/test_data/self_generated/death_pos'
    #    neg_testfile_name = 'data/test_data/self_generated/death_neg'
    #    with open(pos_testfile_name, 'r') as pos_testfile:
    #        for line in pos_testfile:
    #            test_positive_examples.append(line)
    #    with open(neg_testfile_name, 'r') as neg_testfile:
    #        for line in neg_testfile:
    #            test_negative_examples.append(line)
    #    test_examples = test_positive_examples + test_negative_examples
    #    test_labels = ([1 for e in test_positive_examples] +
    #                   [0 for e in test_negative_examples])



    predicted_labels = test(job_id,test_examples,test_labels)

    print "predicted_labels"
    print predicted_labels
    sys.stdout.flush()

    precision, recall, f1 = computeScores(predicted_labels, test_labels)


    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []

    for example, label in zip(
            test_positive_examples,
            predicted_labels[0:len(test_positive_examples)]):
        if label == 1:
            true_positives.append(example)
        else:
            false_negatives.append(example)

    for example, label in zip(
            test_negative_examples,
            predicted_labels[len(test_positive_examples):]):
        if label == 1:
            false_positives.append(example)
        else:
            true_negatives.append(example)


    return (true_positives,
            false_positives,
            true_negatives,
            false_negatives,
            [precision, recall, f1])



def compute_performance_on_test_set(job_id, task_ids, experiment,
                                    training_positive_examples = [], 
                                    training_negative_examples =[]):

    number_of_times_to_test = 3
    precisions = []
    recalls = []
    f1s = []

    #if experiment.test_set == 'death':
    #    test_set_index = 3

    for i in range(number_of_times_to_test):
        retrain(job_id, ['all'], task_ids, 
                training_positive_examples, 
                training_negative_examples)
        job = Job.objects.get(id=job_id)
        (true_positives, false_positives,
         true_negatives, false_negatives,
         [precision, recall, f1]) = test_on_held_out_set(
             job_id, ['all'], job.test_set)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        #curframe = inspect.currentframe()                                      
        #calframe = inspect.getouterframes(curframe, 2)                         
        #caller =  calframe[1][3]                                               

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    return avg_precision, avg_recall, avg_f1

