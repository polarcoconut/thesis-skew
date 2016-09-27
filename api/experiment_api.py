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

import time
from api.mturk_connection.mturk_connection import MTurk_Connection
from api.mturk_connection.mturk_connection_real import MTurk_Connection_Real
from api.mturk_connection.mturk_connection_sim import MTurk_Connection_Sim
import cPickle

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
    
    for i in range(experiment.num_runs):


        job = Job(task_information = experiment.task_information,
                  num_training_examples_in_model = -1,
                  current_hit_ids = [],
                  checkpoints = {},
                  status = 'Running',
                  control_strategy = experiment.control_strategy,
                  experiment_id = experiment_id)

        #job.model_file.put("placeholder")
        #job.model_meta_file.put("placeholder")
        job.save()

        job_id = str(job.id)

        mturk_connection = cPickle.dumps(
            MTurk_Connection_Sim(experiment_id, job_id))
        job.mturk_connection.put(mturk_connection)
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

        (task_information, budget) = pickle.loads(experiment.task_information)
        
        return [task_information, budget, experiment.job_ids,
                experiment.num_runs,
                experiment.learning_curves,
                experiment.control_strategy]



experiment_analyze_parser = reqparse.RequestParser()
experiment_analyze_parser.add_argument('experiment_id', type=str, required=True)

class ExperimentAnalyzeApi(Resource):
    def get(self):

        args = experiment_analyze_parser.parse_args()
        experiment_id = args['experiment_id']

        experiment = Experiment.objects.get(id=experiment_id)

        precisions = []
        recalls = []
        f1s = []
        for job_id in experiment.job_ids:
            learning_curve = experiment.learning_curves[job_id]

            print len(learning_curve)
            #This initialization should only occur once
            if len(precisions) == 0:
                precisions = [[] for i in range(len(learning_curve))]
                recalls = [[] for i in range(len(learning_curve))]
                f1s = [[] for i in range(len(learning_curve))]
                                   
            for point_index, point in zip(range(len(learning_curve)),
                                          learning_curve):
                precision, recall, f1 = point
                precisions[point_index].append(precision)
                recalls[point_index].append(recall)
                f1s[point_index].append(f1)
            
        precisions = [
            float(sum(numbers)) / len(numbers) for numbers in precisions]
        recalls = [
            float(sum(numbers)) / len(numbers) for numbers in recalls]
        f1s = [
            float(sum(numbers)) / len(numbers) for numbers in f1s]
        number_of_labels = [50 * i for i in range(len(f1s))]


        precision_curve = []
        recall_curve =[]
        f1_curve = []
        
        for x,precision,recall,f1 in zip(number_of_labels,
                                         precisions,
                                         recalls,
                                         f1s):
            precision_curve.append({"x" : x, "y" :precision})
            recall_curve.append({"x" : x, "y" : recall}) 
            f1_curve.append({"x" : x, "y" : f1}) 

        return [precision_curve, recall_curve, f1_curve]
