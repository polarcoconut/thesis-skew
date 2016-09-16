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
            0: ['data/training_data/data_for_simulation/death_positives'],
            1: ['data/training_data/data_for_simulation/death_negatives']}
        files_for_simulation = pickle.dumps(files_for_simulation)
        
        experiment = Experiment(
            job_ids = [],
            task_information = pickle.dumps((task_information, budget)),
            num_runs = num_runs,
            control_strategy = control_strategy,
            test_set = 3,
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

    print "Train a classifier using angli's data."
    sys.stdout.flush()

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

    for i in range(experiment.num_runs):


        job = Job(task_information = experiment.task_information,
                  num_training_examples_in_model = -1,
                  current_hit_ids = [],
                  checkpoints = {},
                  status = 'Running',
                  control_strategy = experiment.control_strategy,
                  experiment_id = experiment_id)
        job.save()

        job_id = str(job.id)

        mturk_connection = cPickle.dumps(
            MTurk_Connection_Sim
            (experiment_id, job_id, model_file_name, vocabulary))
        job.mturk_connection.put(mturk_connection)
        job.save()

        
        experiment.job_ids.append(job_id)
        experiment.learning_curves[job_id] = []
        experiment.save()

        gather(task_information, budget, job_id)

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
