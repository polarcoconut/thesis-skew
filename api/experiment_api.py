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
        task_ids_for_simulation = {0: ['57cf315e73ee160009a7da56'],
                                   1: ['57cfa7a3a4b38d000cd0d6e5']}

        experiment = Experiment(
            job_ids = [],
            task_information = pickle.dumps((task_information, budget))
            num_runs = num_runs,
            control_strategy = control_strategy,
            test_set = 3,
            task_ids_for_simulation = task_ids_for_simulation,
            precisions = []
            recalls = []
            fscores = [])

        experiment_id = str(experiment.id)
        
        run_experiment.delay(experiment_id)
        
        return redirect(url_for(
            'experiment_status',  
            experiment_id = experiment_id))

@app.celery.task(name='experiment')
def run_experiment(experiment_id):

    experiment = Experiment.objects.get(id = experiment_id)        

    (task_information, budget) = pickle.loads(experiment.task_information)
    
    while i in range(experiment.num_runs):
        
        job = Job(task_information = task_information,
                  num_training_examples_in_model = -1,
                  current_hit_ids = [],
                  checkpoints = {},
                  status = 'Running',
                  control_strategy = experiment.control_strategy,
                  experiment_id = experiment_id)

        job.save()
        job_id = str(job.id)

        experiment.job_ids.append(job_id)
        experiment.save()
        
        gather(task_information, budget, job_id)
        
        checkpoint = getLatestCheckpoint(job_id)
        
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        relations = ['nationality', 'born', 'lived', 'died', 'travel']
        amount_of_data = [1898, 496, 3897, 1493, 1992]
        testfile_name = 'data/test_data/test_strict_new_feature'
        (test_labels, test_features, test_examples,
         test_positive_examples,
         test_negative_examples) = parse_angli_test_data(
             testfile_name, [], experiment.test_set)
            
        vocabulary = pickle.loads(job.vocabulary)

        predicted_labels = test_cnn(
            test_examples,
            test_labels,
            write_model_to_file(job_id),
            vocabulary)


        precision, recall, f1 = computeScores(predicted_labels, test_labels)

        experiment.precisions.append(precision)
        experiment.recalls.append(recall)
        experiment.fscores.append(f1)

        experiment.save()
        
        


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
                experiment.precision,
                experiment.recalls,
                experiment.fscores,
                experiment.control_strategy]
