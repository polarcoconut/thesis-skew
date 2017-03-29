from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import gather, restart, gather_status
from train_simulate import gather_sim
import sys
import uuid
from schema.job import Job
from schema.experiment import Experiment
from schema.gold_extractor import Gold_Extractor
from util import parse_task_information, retrain, getLatestCheckpoint, split_examples, parse_answers, write_model_to_file
from crowdjs_util import get_task_data
from ml.extractors.cnn_core.parse import parse_angli_test_data
from ml.extractors.cnn_core.train import train_cnn
from api.s3_util import insert_connection_into_s3, generate_dataset
import time
from api.mturk_connection.mturk_connection import MTurk_Connection
from api.mturk_connection.mturk_connection_real import MTurk_Connection_Real
from api.mturk_connection.mturk_connection_sim import MTurk_Connection_Sim
from api.mturk_connection.mturk_connection_super_sim import MTurk_Connection_Super_Sim
import cPickle
from ml.extractors.cnn_core.parse import parse_angli_test_data
from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores
import requests
from random import sample, shuffle
import pprint
import sys, os, traceback, time, re
import inspect
import psutil

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
experiment_parser.add_argument('gpu_device_string', type=str, required=True)


class ExperimentApi(Resource):
    def post(self):
      

        #Clean up code
        #for job in Job.objects():
        #    if not job.status == 'Finished':
        #        job.delete()
        #return None
            
        
        args = experiment_parser.parse_args()
        task_information = parse_task_information(args)
                
        budget = int(args['budget'])
        control_strategy = args['control_strategy']
        num_runs = args['num_runs']
        gpu_device_string = args['gpu_device_string']

        event_name = args['event_name'].lower()

        #ratios = [1]
        ratios = [1,2,3,5,9,49,99]

        for num_of_negatives_per_positive in ratios:
 

            experiment = Experiment(
                job_ids = [],
                task_information = pickle.dumps((task_information, budget)),
                num_runs = num_runs,
                control_strategy = control_strategy,
                control_strategy_configuration = '%s,%s,%s,%s' % (
                    app.config['UCB_EXPLORATION_CONSTANT'], 
                    app.config['MODEL'],
                    num_of_negatives_per_positive,
                    app.config['NUM_NEGATIVES_PER_POSITIVE']), 
                learning_curves= {},
                gpu_device_string = gpu_device_string,
                status = 'Running',
                dataset_skew = num_of_negatives_per_positive)

            #Compute an upper bound on performance
            #compute_upper_bound.delay(experiment.gold_extractor, 
            #                          experiment.test_set)
            #raise Exception

            experiment.save()

            experiment_id = str(experiment.id)

            #run_experiment.delay(experiment_id)
            run_experiment(experiment_id, event_name)

        return redirect(url_for(
            'experiment_status',  
            experiment_id = experiment_id))


#@app.celery.task(name='experiment')
def run_experiment(experiment_id, event_name):

    experiment = Experiment.objects.get(id = experiment_id)        

    (task_information, budget) = pickle.loads(experiment.task_information)


    #if len(Gold_Extractor.objects(name=experiment.gold_extractor)) < 1:
    #    train_gold_extractor(experiment.gold_extractor)
    

    #Test how good the gold extractor is on the test set.
    
    """
    print "TESTING HOW GOOD THE GOLD EXTRACTOR IS"
    sys.stdout.flush()

    #testfile_name = 'data/test_data/test_strict_new_feature'                
    #(test_labels, test_features, test_examples,                             
    # test_positive_examples,                                                
    # test_negative_examples) = parse_angli_test_data(                       
    #     testfile_name, [], experiment.test_set)

    corpus = str(requests.get(
        experiment.unlabeled_corpus).content).split('\n')
    test_examples = []
    test_labels = []
    for sentence in corpus:
        test_examples.append(sentence)
        test_labels.append(0)

    gold_extractor = Gold_Extractor.objects.get(name=experiment.gold_extractor)
    model_file_name = write_model_to_file(
        gold_extractor = gold_extractor.name)    
    vocabulary = cPickle.loads(str(gold_extractor.vocabulary))
    predicted_labels, label_probabilities = test_cnn(test_examples,
                                test_labels,
                                model_file_name,
                                vocabulary)
    
    print "PROPORTION OF POSITIVES IN DATA"
    num_positives = 0.0
    for predicted_label in predicted_labels:
        if predicted_label == 1:
            num_positives += 1
    print num_positives / len(predicted_labels)
    sys.stdout.flush()

    precision, recall, f1 = computeScores(predicted_labels, test_labels)

    print "PRECISION, RECALL, F1"
    print precision, recall, f1
    sys.stdout.flush()
    
    raise Exception
    """

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
                  experiment_id = experiment_id,
                  gpu_device_string = experiment.gpu_device_string,
                  dataset_skew = experiment.dataset_skew)

        #job.model_file.put("placeholder")
        #job.model_meta_file.put("placeholder")
        job.save()

        job_id = str(job.id)
        experiment.job_ids.append(job_id)
        experiment.learning_curves[job_id] = []
        experiment.save()


        start_job.delay(experiment_id, experiment, 
                        job_id, job,
                        task_information, budget,
                        event_name)
        print "DONE SPAWNING JOB NUMBER"
        print i
        sys.stdout.flush()
        #release_lock()
        

@app.celery.task(name='start-job')
def start_job(experiment_id, experiment, 
              job_id, job,
              task_information, budget,
              event_name):

    #Only set the dataset after you start the job.
    if event_name == 'death':
        unlabeled_corpus = app.config['TACKBP_NW_09_CORPUS_URL']
        test_set_index = "3"
        gold_extractor_name = 'death'
        files_for_simulation = {
            0: ['https://s3-us-west-2.amazonaws.com/extremest-extraction-data-for-simulation/death_positives' % event_name],
            1: ['https://s3-us-west-2.amazonaws.com/extremest-extraction-data-for-simulation/death_negatives' % event_name]}
    else:

        (positive_crowd_examples_url, 
         negative_crowd_examples_url,
         unlabeled_corpus_url, 
         labeled_corpus_url,
         positive_testing_examples_url, 
         negative_testing_examples_url) = generate_dataset(
             event_name, 
             experiment.dataset_skew) 
        
        files_for_simulation = {0 : [positive_crowd_examples_url],
                                1 : [negative_crowd_examples_url]}
        unlabeled_corpus = unlabeled_corpus_url
        gold_extractor_name = labeled_corpus_url
        test_set_index = (positive_testing_examples_url + "\t" +
                          negative_testing_examples_url)


    print gold_extractor_name
    print unlabeled_corpus
    sys.stdout.flush()

    job.test_set = test_set_index
    job.gold_extractor = gold_extractor_name
    job.files_for_simulation = pickle.dumps(files_for_simulation)
    job.unlabeled_corpus = unlabeled_corpus
    job.save()

    ##########
    # save the mturk connection 
    ##########

    #mturk_connection_url = insert_connection_into_s3(cPickle.dumps(
    #    MTurk_Connection_Sim(experiment_id, job_id)))
    #mturk_connection_url = insert_connection_into_s3(cPickle.dumps(
    #    MTurk_Connection_Super_Sim(experiment_id, job_id)))
    #job.mturk_connection = mturk_connection_url
    #job.save()


    lock_key = job.id        
    acquire_lock = lambda: app.redis.setnx(lock_key, '1')
    release_lock = lambda: app.redis.delete(lock_key)

    acquire_lock()
    #gather(task_information, budget, job_id)
    print "SPAWNING JOB"
    sys.stdout.flush()
    #app.celery.send_task("gathersim", args = [
    #    task_information, budget, job_id,
    #    MTurk_Connection_Super_Sim(experiment_id, job_id)])

    try:
        gather_sim(task_information, budget, job_id,
                   MTurk_Connection_Super_Sim(experiment_id, job_id))
    except Exception:
        print "Exception:"
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        experiment.exceptions.append(traceback.format_exc())
        job.exceptions.append(traceback.format_exc())

        local_variables = inspect.trace()[-1][0].f_locals
        pprint.pprint(local_variables)
        experiment.exceptions.append(pprint.pformat(local_variables))
        experiment.save()
        job.exceptions.append(pprint.pformat(local_variables))
        job.save()

        print '-'*60
        print "Killing background processes"

        celery_processes = []
        for proc in psutil.process_iter():
            try:
                if proc.name() == "celery":
                    celery_processes.append(proc)
            except psutil.NoSuchProcess:
                pass

        celery_processes = sorted(celery_processes,
                                  key= lambda proc: proc.create_time,
                                  reverse=True)

        print [(proc.name(),
                proc.create_time) for proc in celery_processes]

        for proc in celery_processes:
            if proc.pid == os.getpid():
                continue
                proc.kill()
    finally:
        release_lock()

    #Check if the experiment is done.
    for job_id in experiment.job_ids:
        job = Job.objects.get(id = job_id)
        if job.status == 'Running':
            return None
    
    experiment.status = 'Finished'
    experiment.save()
    
    return None
    
@app.celery.task(name='compute_upper_bound')
def compute_upper_bound(gold_extractor, test_set_index):

    test_examples = []
    test_labels = []

    tackbp_newswire_corpus = str(requests.get(
        app.config['TACKBP_NW_09_CORPUS_URL']).content).split('\n')
 
    for sentence in tackbp_newswire_corpus:
        test_examples.append(sentence)
        test_labels.append(0)

    gold_extractor = Gold_Extractor.objects.get(name=gold_extractor)
    model_file_name = write_model_to_file(
        gold_extractor = gold_extractor.name)    
    vocabulary = cPickle.loads(str(gold_extractor.vocabulary))
    predicted_labels, label_probabilities = test_cnn(test_examples,
                                test_labels,
                                model_file_name,
                                vocabulary)
    
    positive_examples = []
    negative_examples = []
    for i in range(len(predicted_labels)):
        predicted_label = predicted_labels[i]
        example = test_examples[i]
        if predicted_label == 1:
            positive_examples.append(example)
        else:
            negative_examples.append(example)



    num_positive_examples_list  = [1000, 666, 500, 400, 333, 250]
    #num_positive_examples_list  = [1000]
    f1_results_raw = {}
    precision_results_raw = {}
    recall_results_raw = {}
    f1_results = {}
    precision_results = {}
    recall_results = {}
    num_trials = 5

    
    for num_positive_examples in num_positive_examples_list:
        print "TRYING WITH NUMBER OF POSITIVE EXAMPLES"
        print num_positive_examples
        sys.stdout.flush()

        f1_results_raw[num_positive_examples] = []
        precision_results_raw[num_positive_examples] = []
        recall_results_raw[num_positive_examples] = []
        for num_trial in range(num_trials):
            num_negative_examples = 2000 - num_positive_examples
            
            selected_examples = []
            expected_labels = []

            #if len(positive_examples) < num_positive_examples_to_label:
            #    selected_examples += positive_examples
            #    expected_labels += [1 for i in range(len(positive_examples))]
            #    selected_examples += sample(
            #        negative_examples,
            #        app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(positive_examples)\
                #    )
            #       expected_labels += [0 for i in range(
            #           app.config['CONTROLLER_LABELING_BATCH_SIZE']-
            #           len(positive_examples))]
            
            #   elif len(negative_examples) < num_negative_examples_to_label:
            #       selected_examples += negative_examples
            #       expected_labels += [0 for i in range(len(negative_examples))]
            #       selected_examples += sample(
            #           positive_examples,
            #           app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(negative_examples)\
            #       )
            #       expected_labels += [1 for i in range(
            #           app.config['CONTROLLER_LABELING_BATCH_SIZE']-
            #           len(negative_examples))]
            #   else:
        
            selected_examples += sample(positive_examples,
                                        num_positive_examples)
            expected_labels += [1 for i in range(num_positive_examples)]
            selected_examples += sample(negative_examples,
                                        num_negative_examples)
            expected_labels += [0 for i in range(num_negative_examples)]
        
            #shuffle(selected_examples)

            model_file_name, vocabulary = train_cnn(
                selected_examples, expected_labels,
                '/gpu:0')
            
            #(model_url, model_meta_url,
            # model_key, model_meta_key) = insert_model_into_s3(
            #    model_file_name,
            #    "{}.meta".format(model_file_name))
            
            
            testfile_name = 'data/test_data/test_strict_new_feature'   
            (test_labels, test_features, test_examples,                        
             test_positive_examples,                                          
             test_negative_examples) = parse_angli_test_data(
                 testfile_name, [], test_set_index)           
            predicted_labels, label_probabilities = test_cnn(
                test_examples, test_labels,
                model_file_name,
                vocabulary)

            precision, recall, f1 = computeScores(predicted_labels, 
                                                  test_labels)
            f1_results_raw[num_positive_examples].append(f1)
            recall_results_raw[num_positive_examples].append(recall)
            precision_results_raw[num_positive_examples].append(precision)

        f1_results[num_positive_examples] = [
            np.mean(f1_results_raw[num_positive_examples]),
            np.std(f1_results_raw[num_positive_examples])]
        precision_results[num_positive_examples] = [
            np.mean(precision_results_raw[num_positive_examples]),
            np.std(precision_results_raw[num_positive_examples])]
        recall_results[num_positive_examples] = [
            np.mean(recall_results_raw[num_positive_examples]),
            np.std(recall_results_raw[num_positive_examples])]
        

    print "RESULTS"
    print f1_results_raw
    print precision_results_raw
    print recall_results_raw
    print f1_results
    print precision_results
    print recall_results

    upperbound_file = open('upperbounds', 'w')
    upperbound_file.write(pprint.pformat(f1_results_raw))
    upperbound_file.write('\n')
    upperbound_file.write(pprint.pformat(precision_results_raw))
    upperbound_file.write('\n')
    upperbound_file.write(pprint.pformat(recall_results_raw))
    upperbound_file.write('\n')
    upperbound_file.write(pprint.pformat(f1_results))
    upperbound_file.write('\n')
    upperbound_file.write(pprint.pformat(precision_results))
    upperbound_file.write('\n')
    upperbound_file.write(pprint.pformat(recall_results))
    upperbound_file.write('\n')

    
    
    return [f1_results_raw, precision_results_raw, recall_results_raw,
            f1_results, precision_results, recall_results]



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
         [0 for e in training_negative_examples]),
        '/gpu:0')

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

        predicted_precisions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        predicted_recalls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        predicted_f1s = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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
        

        if len(predicted_precisions) > 14:
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


all_experiment_analyze_parser = reqparse.RequestParser()
all_experiment_analyze_parser.add_argument('domain', 
                                           type=str, required=True)
all_experiment_analyze_parser.add_argument('classifier', 
                                           type=str, required=True)

#all_experiment_analyze_parser.add_argument('job_ids', 
#                                           type=str, action='append')

class AllExperimentAnalyzeApi(Resource):
    def get(self):

        print "Constructing Graphs"
        sys.stdout.flush()

        args = all_experiment_analyze_parser.parse_args()
        selected_domain = args['domain']
        selected_classifier = args['classifier']
        #job_ids = args['job_ids']

        #experiment = Experiment.objects


        #precision_curves = []
        #recall_curves = []
        #f1_curves = []

        strategies_to_include = ['Seed-Bounded-Ratio','Round-Robin-Bounded-Ratio','Label-Only-Bounded-Ratio', 'Label-Only']
        
        
        strategy_names = {
            'seed3' : 'Seed-Bounded-Ratio',
            'round-robin-constant-ratio' : 'Round-Robin-Bounded-Ratio',
            'label-only-constant-ratio' : 'Label-Only-Bounded-Ratio',
            'label-only' : 'Label-Only'}

        strategy_indexes = {}
        curve_labels = "Skew (Number of Negatives Per Positive)"
        line_item = ""

        current_index = 0
        for strategy in strategies_to_include:
            strategy_indexes[strategy] = current_index
            current_index += 1
            line_item += ",,"
            curve_labels += (",%s" % strategy)

        curve_labels += "\n"
        line_item = line_item[0:-1]
        line_item += "\n"

        precision_curve = curve_labels
        recall_curve  = curve_labels
        f1_curve = curve_labels
        


        for experiment in Experiment.objects:

            experiment_domain = pickle.loads(experiment.task_information)[0][0]
            experiment_csc = experiment.control_strategy_configuration.split(
                ',')

            #print "Selected Domain"
            #print selected_domain
            #print experiment_domain
            #print pickle.loads(experiment.task_information)
            #sys.stdout.flush()

            if not experiment_domain.lower() == selected_domain.lower():
                continue

            if len(experiment_csc) >= 3:
                continue
            if len(experiment_csc) >= 2:
                print "Experiment configuration"
                print experiment_csc
                sys.stdout.flush()
                
                experiment_classifier = experiment_csc[1]
            else:
                experiment_classifier = 'cnn'

                
            if not (experiment_classifier.lower()==
                    selected_classifier.lower()):
                continue



            [precisions_avgs, recalls_avgs, f1s_avgs, 
             precisions_stds, recalls_stds, f1s_stds,
             costSoFars_avgs, costSoFars_std] = get_average_curve(
                 experiment.id)

            x_axis = costSoFars_avgs

            if not str(experiment.control_strategy) in strategy_names:
                continue
            strategy_key = strategy_names[str(experiment.control_strategy)]
            starting_index = strategy_indexes[strategy_key] * 2 

            for (x,precision_avg,recall_avg,
                 f1_avg,precision_std,recall_std,f1_std) in zip(
                     x_axis,
                     precisions_avgs, recalls_avgs, f1s_avgs,
                     precisions_stds, recalls_stds, f1s_stds):
                
                precision_curve += (
                    ("%f," % x) + 
                    line_item[0:starting_index] +
                    ("%f" % precision_avg) +
                    line_item[starting_index:starting_index+1] +
                    ("%f" % precision_std) +
                    line_item[starting_index+1:])
                recall_curve += (
                    ("%f," % x) + 
                    line_item[0:starting_index] +
                    ("%f" % recall_avg) +
                    line_item[starting_index:starting_index+1] +
                    ("%f" % recall_std) +
                    line_item[starting_index+1:])
                f1_curve += (
                    ("%f," % x) + 
                    line_item[0:starting_index] +
                    ("%f" % f1_avg) +
                    line_item[starting_index:starting_index+1] +
                    ("%f" % f1_std) +
                    line_item[starting_index+1:])

                
        return [precision_curve, recall_curve, f1_curve]

@app.celery.task(name='get_num_examples_labeled')
def get_num_examples_labeled(experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)

    if ((not experiment.control_strategy == 'round-robin-constant-ratio') and
        (not experiment.control_strategy == 'round-robin-constant-ratio-random-labeling')):
        return None

    num_labeled_examples = []
    num_label_actions = []
    job_ids = experiment.job_ids

    #Count the number of positives found by the label actions
    for job_id in job_ids:
        checkpoint = getLatestCheckpoint(job_id)        
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        num_labeled_examples_per_job = 0
        num_label_actions_per_job = 0
        for task_id, task_category_id in zip(task_ids, task_categories):
            if task_category_id == 2:
                answers = parse_answers(task_id, task_category_id)
                new_examples, new_labels = answers
                num_labeled_examples_per_job += len(new_examples)
                num_label_actions_per_job += 1

        num_label_actions.append(num_label_actions_per_job)
        num_labeled_examples.append(num_labeled_examples_per_job)
    
    print "HERE IS THE AVERAGE NUMBER OF EXAMPLES LABELED FOR STRATEGY"
    print experiment.control_strategy
    print num_labeled_examples
    print np.mean(num_labeled_examples)
    print num_label_actions
    print np.mean(num_label_actions)
    sys.stdout.flush()

def get_average_curve(experiment_id):

    experiment = Experiment.objects.get(id=experiment_id)
    job_ids = experiment.job_ids
    precisions = []
    recalls = []
    f1s = []
    
    print "GETTING THE AVERAGE CURVE FOR CONTROL STRATEGY"
    print experiment.control_strategy


    #get_num_examples_labeled.delay(experiment_id)
 
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
    costSoFars =  [[] for i in range(len_longest_curve)]
        
    for job_id in job_ids:
        learning_curve = experiment.learning_curves[job_id]           
        for point_index, point in zip(range(len(learning_curve)),
                                      learning_curve):
            task_id, precision, recall, f1, action, costSoFar = point
            precisions[point_index].append(precision)
            recalls[point_index].append(recall)
            f1s[point_index].append(f1)
            costSoFars[point_index].append(costSoFar)

    precisions_avgs = [np.mean(numbers) for numbers in precisions]
    recalls_avgs = [np.mean(numbers) for numbers in recalls]
    f1s_avgs = [np.mean(numbers) for numbers in f1s]
    costSoFars_avgs = [np.mean(numbers) for numbers in costSoFars]

    precisions_stds = [np.std(numbers) / 
                       sqrt(len(numbers)) for numbers in precisions]
    recalls_stds = [np.std(numbers) /
                    sqrt(len(numbers)) for numbers in recalls]
    f1s_stds = [np.std(numbers) / 
                sqrt(len(numbers)) for numbers in f1s]
    
    costSoFars_stds = [np.std(numbers) / 
                       sqrt(len(numbers)) for numbers in costSoFars]
    

    return [precisions_avgs, recalls_avgs, f1s_avgs, 
            precisions_stds, recalls_stds, f1s_stds,
            costSoFars_avgs, costSoFars_stds]



def get_average_aoc(experiment_id):

    experiment = Experiment.objects.get(id=experiment_id)
    job_ids = experiment.job_ids
    precisions = []
    recalls = []
    f1s = []
    
    print "GETTING THE AVERAGE CURVE FOR CONTROL STRATEGY"
    print experiment.control_strategy


 
    #len_longest_curve = 0
    #first figure out the longest curve
    

    #for job_id in job_ids:
    #    learning_curve = experiment.learning_curves[job_id]
    #    if len(learning_curve) > len_longest_curve:
    #        len_longest_curve = len(learning_curve)
    #        print len(learning_curve)

    precision_curve_aocs = []
    recall_curve_aocs = []
    f1_curve_aocs = []

    for job_id in job_ids:
        learning_curve = experiment.learning_curves[job_id]           
        precisions = []
        recalls = []
        f1s = []
        costSoFars = []

        for point in learning_curve:
            task_id, precision, recall, f1, action, costSoFar = point
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            #costSoFars.append(costSoFar)
        
        #if (experiment.control_strategy == 'label-only' or
        #    experiment.control_strategy == 'label-only-constant-ratio'):
        #    costSoFars = [7,14,20,27,34,40,47,54,60,67,67]
        #elif experiment.control_strategy == 'round-robin-constant-ratio':
        #    costSoFars = [3,5,7,9,11,13,15,18,20,23,23]
        #elif experiment.control_strategy == 'seed3':
        #    costSoFars = [2,3,8,15,22,28,35,42,48,55,55]

        if (experiment.control_strategy == 'label-only' or
            experiment.control_strategy == 'label-only-constant-ratio'):
            costSoFars = [7,14,20]
            precisions = precisions[0:3]
            recalls = recalls[0:3]
            f1s = f1s[0:3]
        elif experiment.control_strategy == 'round-robin-constant-ratio':
            costSoFars = [3,5,7,9,11,13,15,18,20,23,23]
            precisions = precisions[0:11]
            recalls = recalls[0:11]
            f1s = f1s[0:11]
        elif experiment.control_strategy == 'seed3':
            costSoFars = [2,3,8,15,22]
            precisions = precisions[0:5]
            recalls = recalls[0:5]
            f1s = f1s[0:5]


        precision_curve_aocs.append(np.trapz(precisions, costSoFars))
        recall_curve_aocs.append(np.trapz(recalls, costSoFars))
        f1_curve_aocs.append(np.trapz(f1s, costSoFars))

    precision_aoc_avg = np.mean(precision_curve_aocs)
    recall_aoc_avg = np.mean(recall_curve_aocs)
    f1_aoc_avg = np.mean(f1_curve_aocs)

    precision_aoc_std = (np.std(precision_curve_aocs) / 
                         sqrt(len(precision_curve_aocs)))
    recall_aoc_std = (np.std(recall_curve_aocs)  / 
                      sqrt(len(recall_curve_aocs)))
    f1_aoc_std = (np.std(f1_curve_aocs)  / 
                      sqrt(len(f1_curve_aocs)))

    return [precision_aoc_avg, recall_aoc_avg, f1_aoc_avg,
            precision_aoc_std, recall_aoc_std, f1_aoc_std]


class SkewAnalyzeApi(Resource):
    def get(self):

        print "Constructing Graphs"
        sys.stdout.flush()

        args = all_experiment_analyze_parser.parse_args()
        selected_domain = args['domain']
        selected_classifier = args['classifier']


        strategies_to_include = ['Seed-Bounded-Ratio','Round-Robin-Bounded-Ratio','Label-Only-Bounded-Ratio', 'Label-Only']
        
        
        strategy_names = {
            'seed3' : 'Seed-Bounded-Ratio',
            'round-robin-constant-ratio' : 'Round-Robin-Bounded-Ratio',
            'label-only-constant-ratio' : 'Label-Only-Bounded-Ratio',
            'label-only' : 'Label-Only'}

        strategy_indexes = {}
        curve_labels = "Skew (Number of Negatives Per Positive)"
        line_item = ""

        current_index = 0
        for strategy in strategies_to_include:
            strategy_indexes[strategy] = current_index
            current_index += 1
            line_item += ",,"
            curve_labels += (",%s" % strategy)

        curve_labels += "\n"
        line_item = line_item[0:-1]
        line_item += "\n"

        precision_curve = curve_labels
        recall_curve  = curve_labels
        f1_curve = curve_labels

        """
        precision_curve = "Skew (Number of Negatives Per Positive),Seed,Round-Robin-Crowd-Negatives,Round-Robin-Random-Negatives,Round-Robin-Constant-Ratio,Label-Only-Constant-Ratio,Round-Robin-Constant-Ratio-Random-Labeling\n"
        recall_curve = "Skew (Number of Negatives Per Positive),Seed,Round-Robin-Crowd-Negatives,Round-Robin-Random-Negatives,Round-Robin-Constant-Ratio,Label-Only-Constant-Ratio,Round-Robin-Constant-Ratio-Random-Labeling\n"
        f1_curve = "Skew (Number of Negatives Per Positive),Seed,Round-Robin-Crowd-Negatives,Round-Robin-Random-Negatives,Round-Robin-Constant-Ratio,Label-Only-Constant-Ratio,Round-Robin-Constant-Ratio-Random-Labeling\n"
        """

        for experiment in Experiment.objects:

            experiment_domain = pickle.loads(experiment.task_information)[0][0]
            experiment_csc = experiment.control_strategy_configuration.split(
                ',')

            #print "Selected Domain"
            #print selected_domain
            #print experiment_domain
            #print pickle.loads(experiment.task_information)
            #sys.stdout.flush()

            if not experiment_domain.lower() == selected_domain.lower():
                continue

            if len(experiment_csc) >= 3:
                print "Experiment configuration"
                print experiment_csc
                sys.stdout.flush()
                
                experiment_classifier = experiment_csc[1]
                experiment_skew = float(experiment_csc[2])
            else:
                continue
                
            if not (experiment_classifier.lower()==
                    selected_classifier.lower()):
                continue

            [precision_aoc, recall_aoc, f1_aoc, 
             precision_std, recall_std, f1_std] = get_average_aoc(
                 experiment.id)

            print "Computed stuff"
            print [precision_aoc, recall_aoc, f1_aoc, 
                   precision_std, recall_std, f1_std]
            sys.stdout.flush()
                
            if not str(experiment.control_strategy) in strategy_names:
                continue
            strategy_key = strategy_names[str(experiment.control_strategy)]

            starting_index = strategy_indexes[strategy_key] * 2 
            precision_curve += (
                ("%f," % experiment_skew) + 
                line_item[0:starting_index] +
                ("%f" % precision_aoc) +
                line_item[starting_index:starting_index+1] +
                ("%f" % precision_std) +
                line_item[starting_index+1:])
            recall_curve += (
                ("%f," % experiment_skew) + 
                line_item[0:starting_index] +
                ("%f" % recall_aoc) +
                line_item[starting_index:starting_index+1] +
                ("%f" % recall_std) +
                line_item[starting_index+1:])
            f1_curve += (
                ("%f," % experiment_skew) + 
                line_item[0:starting_index] +
                ("%f" % f1_aoc) +
                line_item[starting_index:starting_index+1] +
                ("%f" % f1_std) +
                line_item[starting_index+1:])

                

        return [precision_curve, recall_curve, f1_curve]
