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
from util import parse_task_information, retrain, getLatestCheckpoint, split_examples, parse_answers
from crowdjs_util import get_task_data


train_parser = reqparse.RequestParser()
train_parser.add_argument('event_name', type=str, required=True)
train_parser.add_argument('event_definition', type=str, required=True)
train_parser.add_argument('event_pos_example_1', type=str, required=True)
train_parser.add_argument('event_pos_example_1_trigger',
                          type=str, required=True)
train_parser.add_argument('event_pos_example_2', type=str, required=True)
train_parser.add_argument('event_pos_example_2_trigger',
                          type=str, required=True)
train_parser.add_argument('event_pos_example_nearmiss', type=str, required=True)
train_parser.add_argument('event_neg_example',
                          type=str, required=True)
train_parser.add_argument('event_neg_example_nearmiss',
                          type=str, required=True)
train_parser.add_argument('budget', type=str, required=True)
train_parser.add_argument('control_strategy', type=str, required=True)

restart_parser = reqparse.RequestParser()
restart_parser.add_argument('job_id', type=str, required=True)

pause_parser = reqparse.RequestParser()
pause_parser.add_argument('job_id', type=str, required=True)

job_status_parser = reqparse.RequestParser()
job_status_parser.add_argument('job_id', type=str, required=True)




retrain_status_parser = reqparse.RequestParser()
retrain_status_parser.add_argument('job_id', type=str, required=True)


class GatherExtractorApi(Resource):
    def post(self):
        args = train_parser.parse_args()
        task_information = parse_task_information(args)
                
        #task_information = args['task_information']
        budget = int(args['budget'])
        control_strategy = args['control_strategy']
    
        #Generate a random job_id
        job = Job(task_information = pickle.dumps((task_information, budget)),
                  num_training_examples_in_model = -1,
                  current_hit_ids = [],
                  checkpoints = {},
                  status = 'Running',
                  control_strategy = control_strategy)
        
        job.save()
        job_id = str(job.id)
        
        gather.delay(task_information, budget, job_id)
            
        return redirect(url_for(
            'status',  
            job_id = job_id))


class RestartApi(Resource):
    def get(self):
        args = restart_parser.parse_args()
        job_id = args['job_id']

        job = Job.objects.get(id = job_id)
        job.status = 'Running'
        job.save()

        print "Job %s restarted" % job_id
        
        return 1

class PauseApi(Resource):
    def get(self):
        args = pause_parser.parse_args()
        job_id = args['job_id']

        job = Job.objects.get(id = job_id)
        job.status = 'Paused'
        job.save()

        print "Job %s paused" % job_id
        return 1

class JobStatusApi(Resource):
    def get(self):
        args = job_status_parser.parse_args()
        job_id = args['job_id']

        job = Job.objects.get(id = job_id)

        checkpoint = getLatestCheckpoint(job_id)
        (task_information, budget) = pickle.loads(job.task_information)

        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)
        try:
            return [job.status, costSoFar]
        except AttributeError:
            job.status = 'Paused'
            job.save()
            return [job.status, costSoFar]

gather_status_parser = reqparse.RequestParser()
gather_status_parser.add_argument('job_id', type=str, required=False)
gather_status_parser.add_argument('positive_types', required=False,
                                  action='append')
gather_status_parser.add_argument('task_id', required=False,
                                  action='append')
gather_status_parser.add_argument('task_category', required=False,
                                  action='append')

#
# Get the examples that have been collected so far
#
#
class GatherStatusApi(Resource):
    def get(self):
        args = gather_status_parser.parse_args()
        if args['job_id']:
            job_id = args['job_id']
            positive_types = args['positive_types']
            return gather_status(job_id, positive_types)            
        else:
            task_ids = args['task_id']
            task_categories = args['task_category']

            all_positive_examples = []
            all_negative_examples = []

            taboo_words_pickles = []
            print task_ids, task_categories
            sys.stdout.flush()

            for (task_id, task_category) in zip(task_ids, task_categories):
                positive_examples, negative_examples = split_examples(
                    [task_id], [int(task_category)], ['all'], False)
                all_positive_examples += positive_examples
                all_negative_examples += negative_examples

                #Also get taboo words
                taboo_words_pickle = get_task_data(task_id)
                taboo_words_pickles.append(taboo_words_pickle)

            print "taboo word pickles"
            print taboo_words_pickles
            sys.stdout.flush()
            
            return [len(all_positive_examples) + len(all_negative_examples),
                    all_positive_examples,
                    all_negative_examples,
                    taboo_words_pickles]
        

retrain_parser = reqparse.RequestParser()
retrain_parser.add_argument('job_id', type=str, required=True)
retrain_parser.add_argument('positive_types', required=True,
                            action='append')
retrain_parser.add_argument('task_ids_to_train', required=False,
                            action='append')


class RetrainExtractorApi(Resource):
    def get(self):
        args = retrain_parser.parse_args()
        job_id = args['job_id']
        positive_types = args['positive_types']
        task_ids_to_train = args['task_ids_to_train']
        
        job = Job.objects.get(id = job_id)
        job.num_training_examples_in_model = -1
        job.save()
        
        retrain.delay(job_id, positive_types,
                      task_ids_to_train)
        return True


class RetrainStatusApi(Resource):
    def get(self):
        args = retrain_status_parser.parse_args()
        job_id = args['job_id']

        job = Job.objects.get(id=job_id)
        return job.num_training_examples_in_model



