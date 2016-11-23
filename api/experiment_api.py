from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import gather, restart, gather_status
import sys
import uuid
from schema.job import Job
from schema.experiment import Experiment
from schema.gold_extractor import Gold_Extractor
from util import parse_task_information, retrain, getLatestCheckpoint, split_examples, parse_answers, write_model_to_file
from crowdjs_util import get_task_data
from ml.extractors.cnn_core.parse import parse_angli_test_data
from ml.extractors.cnn_core.train import train_cnn
from api.s3_util import insert_connection_into_s3
import time
from api.mturk_connection.mturk_connection import MTurk_Connection
from api.mturk_connection.mturk_connection_real import MTurk_Connection_Real
from api.mturk_connection.mturk_connection_sim import MTurk_Connection_Sim
import cPickle
from ml.extractors.cnn_core.parse import parse_angli_test_data
from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores

import numpy as np

from math import sqrt

experiment_parser = reqparse.RequestParser()
experiment_parser.add_argument('event_name', type=str, required=True)
experiment_parser.add_argument('event_definition', type=str, required=True)
experiment_parser.add_argument('event_pos_example_1', type=str, required=True)
experiment_parser.add_argument('event_pos_example_1_trigger',
                          type=str, required=True)
experiment_parser.add_argument('event_pos_example_2', type=str, required=True)
experiment_parser.add_argument('event_pos_example_2_trigger',
                          type=str, required=True)
experiment_parser.add_argument('event_pos_example_nearmiss', type=str, required=True)
experiment_parser.add_argument('event_neg_example',
                          type=str, required=True)
experiment_parser.add_argument('event_neg_example_nearmiss',
                          type=str, required=True)
experiment_parser.add_argument('budget', type=str, required=True)
experiment_parser.add_argument('control_strategy', type=str, required=True)
experiment_parser.add_argument('num_runs', type=int, required=True)


class ExperimentApi(Resource):
    def post(self):
        args = experiment_parser.parse_args()
        task_information = parse_task_information(args)
                
        budget = int(args['budget'])
        control_strategy = args['control_strategy']
        num_runs = args['num_runs']
        files_for_simulation = {
            0: ['https://s3-us-west-2.amazonaws.com/extremest-extraction-data-for-simulation/death_positives'],
            1: ['https://s3-us-west-2.amazonaws.com/extremest-extraction-data-for-simulation/death_negatives']}
        files_for_simulation = pickle.dumps(files_for_simulation)
         
        experiment = Experiment(
            job_ids = [],
            task_information = pickle.dumps((task_information, budget)),
            num_runs = num_runs,
            control_strategy = control_strategy,
            control_strategy_configuration = '%s' % app.config[
                'UCB_EXPLORATION_CONSTANT'],
            test_set = 3,
            gold_extractor = 'death',
            files_for_simulation = files_for_simulation,
            learning_curves= {})

        experiment.save()
        
        experiment_id = str(experiment.id)
        
        run_experiment.delay(experiment_id)
        
        return redirect(url_for(
            'experiment_status',  
            experiment_id = experiment_id))

@app.celery.task(name='experiment')
def run_experiment(experiment_id):

    experiment = Experiment.objects.get(id = experiment_id)        

    (task_information, budget) = pickle.loads(experiment.task_information)


    if len(Gold_Extractor.objects(name=experiment.gold_extractor)) < 1:
        train_gold_extractor(experiment.gold_extractor)
    

    #Test how good the gold extractor is on the test set.
  
    print "TESTING HOW GOOD THE GOLD EXTRACTOR IS"
    sys.stdout.flush()

    testfile_name = 'data/test_data/test_strict_new_feature'                
    (test_labels, test_features, test_examples,                             
     test_positive_examples,                                                
     test_negative_examples) = parse_angli_test_data(                       
         testfile_name, [], experiment.test_set)

    gold_extractor = Gold_Extractor.objects.get(name=experiment.gold_extractor)
    model_file_name = write_model_to_file(
        gold_extractor = gold_extractor.name)    
    vocabulary = cPickle.loads(str(gold_extractor.vocabulary))
    predicted_labels = test_cnn(test_examples,
                                test_labels,
                                model_file_name,
                                vocabulary)
    
    precision, recall, f1 = computeScores(predicted_labels, test_labels)

    print "PRECISION, RECALL, F1"
    print precision, recall, f1
    sys.stdout.flush()
    


    for i in range(experiment.num_runs):


        job = Job(task_information = experiment.task_information,
                  num_training_examples_in_model = -1,
                  current_hit_ids = [],
                  checkpoints = {},
                  status = 'Running',
                  control_strategy = experiment.control_strategy,
                  control_data = pickle.dumps({0 : [],
                                               1 : [],
                                               2 : []}),
                  logging_data = pickle.dumps([]),
                  experiment_id = experiment_id)

        #job.model_file.put("placeholder")
        #job.model_meta_file.put("placeholder")
        job.save()

        job_id = str(job.id)

        mturk_connection_url = insert_connection_into_s3(cPickle.dumps(
            MTurk_Connection_Sim(experiment_id, job_id)))
        job.mturk_connection = mturk_connection_url
        job.save()

        
        experiment.job_ids.append(job_id)
        experiment.learning_curves[job_id] = []
        experiment.save()

        lock_key = job.id        
        acquire_lock = lambda: app.redis.setnx(lock_key, '1')
        release_lock = lambda: app.redis.delete(lock_key)

        acquire_lock()
        gather(task_information, budget, job_id)
        release_lock()


def train_gold_extractor(name_of_new_gold_extractor):

    pos_training_data_file = open(
        'data/training_data/train_CS_MJ_pos_comb_new_feature_died', 'r')
    
    neg_training_data_file = open(
        'data/training_data/train_CS_MJ_neg_comb_new_feature_died', 'r')
    
    training_positive_examples = []
    training_negative_examples = []
    
    for line in pos_training_data_file:
        training_positive_examples.append(line.split('\t')[11])
    for line in neg_training_data_file:
        training_negative_examples.append(line.split('\t')[11])
        
    # Take a subset of these when testing the code
    #print "Taking a subset of examples for testing purposes"
    #sys.stdout.flush()
    #training_positive_examples = training_positive_examples[0:5]
    #training_negative_examples = training_negative_examples[0:5]
    
    model_file_name, vocabulary = train_cnn(
        training_positive_examples + training_negative_examples,
        ([1 for e in training_positive_examples] +
         [0 for e in training_negative_examples]))

    model_file_handle = open(model_file_name, 'rb')
    #model_binary = model_file_handle.read()

    model_meta_file_handle = open("{}.meta".format(model_file_name), 'rb')
    #model_meta_binary = model_meta_file_handle.read()

    gold_extractor = Gold_Extractor(name=name_of_new_gold_extractor,
                                    vocabulary = cPickle.dumps(vocabulary))

    gold_extractor.model_file.put(model_file_handle)
    gold_extractor.model_meta_file.put(model_meta_file_handle)

    gold_extractor.save()


        
experiment_status_parser = reqparse.RequestParser()
experiment_status_parser.add_argument('experiment_id', type=str, required=True)

class ExperimentStatusApi(Resource):
    def get(self):

        args = experiment_status_parser.parse_args()
        experiment_id = args['experiment_id']

        experiment = Experiment.objects.get(id=experiment_id)

        if not 'control_strategy_configuration' in experiment:
            control_strategy_configuration = "No Configuration"
        else:
            control_strategy_configuration = experiment.control_strategy_configuration
        (task_information, budget) = pickle.loads(experiment.task_information)
        
        return [task_information, budget, experiment.job_ids,
                experiment.num_runs,
                experiment.learning_curves,
                experiment.control_strategy,
                control_strategy_configuration]


experiment_analyze_parser = reqparse.RequestParser()
experiment_analyze_parser.add_argument('experiment_id', type=str, required=True)
experiment_analyze_parser.add_argument('job_ids', type=str, action='append')

class ExperimentAnalyzeApi(Resource):
    def get(self):

        args = experiment_analyze_parser.parse_args()
        experiment_id = args['experiment_id']
        job_ids = args['job_ids']

        experiment = Experiment.objects.get(id=experiment_id)

        precisions = []
        recalls = []
        f1s = []

        
        if job_ids == None:
            job_ids = experiment.job_ids

        len_longest_curve = 0
        #first figure out the longest curve
        for job_id in job_ids:
            learning_curve = experiment.learning_curves[job_id]
            if len(learning_curve) > len_longest_curve:
                len_longest_curve = len(learning_curve)
            print len(learning_curve)
        
        precisions = [[] for i in range(len_longest_curve)]
        recalls = [[] for i in range(len_longest_curve)]
        f1s = [[] for i in range(len_longest_curve)]
        
        for job_id in job_ids:
            learning_curve = experiment.learning_curves[job_id]           
            for point_index, point in zip(range(len(learning_curve)),
                                          learning_curve):
                task_id, precision, recall, f1, action, costSoFar = point
                precisions[point_index].append(precision)
                recalls[point_index].append(recall)
                f1s[point_index].append(f1)

        print job_ids
        print "HUH"

        predicted_precisions = [0,0,0]
        predicted_recalls = [0,0,0]
        predicted_f1s = [0,0,0]
        actions = []
        if len(job_ids) == 1:
            job_id = job_ids[0]
            x_value = 50
            for point in experiment.learning_curves[job_id]:
                task_id, precision, recall, f1, action, costSoFar = point
                #print [x_value, action]
                actions.append([x_value, action])
                x_value += 50

            job = Job.objects.get(id=job_id)
            logging_data = pickle.loads(job.logging_data)
            print logging_data
            print len(logging_data)
            print "HUH"
            sys.stdout.flush()

            if len(logging_data) > 0:
                #x_value = 50
                #print logging_data
                for (best_actions, predictions, values) in logging_data:
                    (predicted_precision, 
                     predicted_recall, predicted_f1) = predictions
                    predicted_precisions.append(predicted_precision)
                    predicted_recalls.append(predicted_recall)
                    predicted_f1s.append(predicted_f1)
                    #x_value += 50
                print "PREDICTIONS"
                print predicted_precisions
                print predicted_recalls
                print predicted_f1s
                sys.stdout.flush()

        #print "Precisions"
        #x_value = 50
        #for precision in precisions:
        #    print x_value
        #    print precision
        #    x_value += 50

        #print "Recalls"
        #x_value = 50
        #for recall in recalls:
        #    print x_value
        #    print recall
        #    x_value += 50

        #print "F1s"
        #x_value = 50
        #for f1 in f1s:
        #    print x_value
        #    print f1
        #    x_value += 50
        #print f1s

        #sys.stdout.flush()

        precisions_avgs = [np.mean(numbers) for numbers in precisions]
        recalls_avgs = [np.mean(numbers) for numbers in recalls]
        f1s_avgs = [np.mean(numbers) for numbers in f1s]

        precisions_stds = [np.std(numbers) / 
                           sqrt(len(numbers)) for numbers in precisions]
        recalls_stds = [np.std(numbers) /
                        sqrt(len(numbers)) for numbers in recalls]
        f1s_stds = [np.std(numbers) / 
                    sqrt(len(numbers)) for numbers in f1s]

        number_of_labels = [50 * i for i in range(1, len(f1s_avgs)+1)]


        #precision_curve = []
        #recall_curve =[]
        #f1_curve = []
        

        if len(logging_data) > 0:
            print "LENGTH OF CURVE"
            print len(precisions_avgs)
            print len(predicted_precisions)
            sys.stdout.flush()

            
            precision_curve = "Number of Labels,Precision,PredictedPrecision\n"
            recall_curve = "Number of Labels,Recall,PredictedRecall\n"
            f1_curve = "Number of Labels,F1,PredictedF1\n"
            
            for (x,precision_avg,recall_avg,
                 f1_avg,precision_std,recall_std,f1_std,
                 predicted_precision, predicted_recall, predicted_f1) in zip(
                     number_of_labels,
                     precisions_avgs, recalls_avgs, f1s_avgs,
                     precisions_stds, recalls_stds, f1s_stds,
                     predicted_precisions, predicted_recalls,
                     predicted_f1s):
                precision_curve += "%d,%f,%f,%f,%f\n" % (
                    x, precision_avg, precision_std, predicted_precision, 0.0)
                recall_curve += "%d,%f,%f,%f,%f\n" % (
                    x, recall_avg, recall_std, predicted_recall, 0.0) 
                f1_curve  += "%d,%f,%f,%f,%f\n" % (
                    x, f1_avg, f1_std, predicted_f1, 0.0) 

            
        else:
            precision_curve = "Number of Labels,Precision\n"
            recall_curve = "Number of Labels,Recall\n"
            f1_curve = "Number of Labels,F1\n"
            
            for (x,precision_avg,recall_avg,
                 f1_avg,precision_std,recall_std,f1_std) in zip(
                     number_of_labels,
                     precisions_avgs, recalls_avgs, f1s_avgs,
                     precisions_stds, recalls_stds, f1s_stds):
                #precision_curve.append({"x" : x, "y" :precision})
                #recall_curve.append({"x" : x, "y" : recall}) 
                #f1_curve.append({"x" : x, "y" : f1}) 
                precision_curve += "%d,%f,%f\n" % (
                    x, precision_avg, precision_std)
                recall_curve += "%d,%f,%f\n" % (x, recall_avg, recall_std) 
                f1_curve  += "%d,%f,%f\n" % (x, f1_avg, f1_std) 
            
            

        print recall_curve
        print "PRECISION CURVE"
        print actions
        sys.stdout.flush()

                
        return [precision_curve, recall_curve, f1_curve, actions,
                predicted_precisions, predicted_recalls, predicted_f1s]
