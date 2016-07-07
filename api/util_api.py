from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import train, restart, gather_status, retrain
import sys
import uuid
import redis
from schema.job import Job

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
        if 'model_meta_file' in timestamps:
            timestamps.remove('model_meta_file')
        if 'vocabulary' in timestamps:
            timestamps.remove('vocabulary')
        if 'num_training_examples_in_model' in timestamps:
            timestamps.remove('num_training_examples_in_model')

        for timestamp in timestamps:
            checkpoints[timestamp] = redis_handle.hmget(job_id, timestamp)

        
        job = Job(task_information = task_information,
                  num_training_examples_in_model = -1,
                  checkpoints = checkpoints)

        job.save()
        
        return str(job.id)
