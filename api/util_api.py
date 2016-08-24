from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
import sys
import uuid
import redis
from schema.job import Job
from schema.ui_test import UI_Test
from controllers import test_controller
from util import parse_task_information, getLatestCheckpoint
from crowdjs_util import upload_questions
from mturk_util import create_hits, delete_hits

arg_parser = reqparse.RequestParser()
arg_parser.add_argument('job_id', type=str, required=True)


class MoveJobsFromRedisToMongoApi(Resource):
    def post(self):
        args = arg_parser.parse_args()
        job_id = args['job_id']

        redis_handle = redis.Redis.from_url(app.config['REDIS_URL'])
        task_information = redis_handle.hmget(job_id, 'task_information')[0]
        timestamps = redis_handle.hkeys(job_id)
        checkpoints = {}
        if 'task_information' in timestamps:
            timestamps.remove('task_information')
        if 'model' in timestamps:
            timestamps.remove('model')
        if 'model_meta' in timestamps:
            timestamps.remove('model_meta')
        if 'model_file' in timestamps:
            timestamps.remove('model_file')
        if 'model_dir' in timestamps:
            timestamps.remove('model_dir')
        if 'model_file_name' in timestamps:
            timestamps.remove('model_file_name')
        if 'model_meta_file' in timestamps:
            timestamps.remove('model_meta_file')
        if 'vocabulary' in timestamps:
            timestamps.remove('vocabulary')
        if 'num_training_examples_in_model' in timestamps:
            timestamps.remove('num_training_examples_in_model')

        for timestamp in timestamps:
            checkpoints[timestamp] = redis_handle.hmget(job_id, timestamp)[0]

        
        job = Job(task_information = task_information,
                  num_training_examples_in_model = -1,
                  checkpoints = checkpoints)

        job.save()
        
        return str(job.id)



class GetJobInfoApi(Resource):

    def get(self):
        args = arg_parser.parse_args()
        job_id = args['job_id']


        job = Job.objects.get(id = job_id)
        
        checkpoint = getLatestCheckpoint(job_id)
        (task_information, budget) = pickle.loads(job.task_information)
        
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        return [task_information, budget, task_ids, task_categories,
                job.control_strategy]

change_budget_arg_parser = reqparse.RequestParser()
change_budget_arg_parser.add_argument('job_id', type=str, required=True)
change_budget_arg_parser.add_argument('new_budget', type=int, required=True)

class ChangeBudgetApi(Resource):
    def post(self):
        args = change_budget_arg_parser.parse_args()
        job_id = args['job_id']
        new_budget = args['new_budget']
        
        job = Job.objects.get(id = job_id)
        (task_information, budget) = pickle.loads(job.task_information)
        job.task_information = pickle.dumps((task_information, new_budget))
        job.save()

        return True
        
        
    

testui_arg_parser = reqparse.RequestParser()
testui_arg_parser.add_argument('event_name', type=str, required=True)
testui_arg_parser.add_argument('event_definition', type=str, required=True)
testui_arg_parser.add_argument('event_pos_example_1', type=str, required=True)
testui_arg_parser.add_argument('event_pos_example_1_trigger',
                          type=str, required=True)
testui_arg_parser.add_argument('event_pos_example_2', type=str, required=True)
testui_arg_parser.add_argument('event_pos_example_2_trigger',
                          type=str, required=True)
testui_arg_parser.add_argument('event_pos_example_nearmiss', type=str, required=True)
testui_arg_parser.add_argument('event_neg_example',
                          type=str, required=True)
testui_arg_parser.add_argument('event_neg_example_nearmiss',
                          type=str, required=True)


class TestGenerateUIApi(Resource):
    def post(self):
        args = testui_arg_parser.parse_args()
        task_information = parse_task_information(args)
        task_category_id, task, num_hits, cost = test_controller(task_information, 0)
        task_id = upload_questions(task)
        hit_ids = create_hits(task_category_id, task_id,
                              num_hits)

        ui_test = UI_Test(task_id = task_id,
                          task_category_id = str(task_category_id),
                          current_hit_ids = hit_ids)
        ui_test.save()
        

        return redirect(url_for('test', task_id = task_id,
                                task_category_id = task_category_id))
 
        
class TestModifyUIApi(Resource):
    def post(self):
        args = testui_arg_parser.parse_args()
        task_information = parse_task_information(args)
        task_category_id, task, num_hits, cost = test_controller(task_information, 1)
        task_id = upload_questions(task)
        hit_ids = create_hits(task_category_id, task_id,
                              num_hits)

        ui_test = UI_Test(task_id = task_id,
                          task_category_id = str(task_category_id),
                          current_hit_ids = hit_ids)
        ui_test.save()
        
        return redirect(url_for('test', task_id = task_id,
                                task_category_id = task_category_id))
 
        
        
class TestLabelUIApi(Resource):
    def post(self):
        args = testui_arg_parser.parse_args()
        task_information = parse_task_information(args)
        task_category_id, task, num_hits, cost = test_controller(task_information, 2)
        task_id = upload_questions(task)
        hit_ids = create_hits(task_category_id, task_id,
                              num_hits)

        ui_test = UI_Test(task_id = task_id,
                          task_category_id = str(task_category_id),
                          current_hit_ids = hit_ids)
        ui_test.save()
        
                          
        return redirect(url_for('test', task_id = task_id,
                                task_category_id = task_category_id))
 

cleanup_arg_parser = reqparse.RequestParser()
cleanup_arg_parser.add_argument('task_id', type=str, required=True)

class CleanUpApi(Resource):
    def get(self):
        args = cleanup_arg_parser.parse_args()
        task_id = args['task_id']
        
        ui_test = UI_Test.objects.get(task_id = task_id)
        
        delete_hits(ui_test.current_hit_ids)

        ui_test.delete()

        return True
        

        
