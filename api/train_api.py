from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for
import json
import string
import pickle
from app import app
from train import train

train_parser = reqparse.RequestParser()
#train_parser.add_argument('event_name', type=str, required=True)
#train_parser.add_argument('event_definition', type=str, required=True)
#train_parser.add_argument('event_good_example_1', type=str, required=True)
#train_parser.add_argument('event_good_example_1_trigger',
#                          type=str, required=True)
#train_parser.add_argument('event_good_example_2', type=str, required=True)
#train_parser.add_argument('event_good_example_2_trigger',
#                          type=str, required=True)
#train_parser.add_argument('event_bad_example_1', type=str, required=True)
#train_parser.add_argument('event_good_negative_example_1',
#                          type=str, required=True)
#train_parser.add_argument('event_bad_negative_example_1',
#                          type=str, required=True)

train_parser.add_argument('task_information', type=str, required=True)
train_parser.add_argument('budget', type=str, required=True)


class TrainExtractorApi(Resource):
    def post(self):
        args = taboo_parser.parse_args()
        #event_name = args['event_name']
        #event_definition = args['event_definition']
        #event_good_example_1 = args['event_good_example_1']
        #event_good_example_2 = args['event_good_example_2']
        #event_bad_example_1 = args['event_bad_example_1']
        task_information = args['task_information']
        budget = int(args['budget'])


        result = app.rq.enqueue(train, task_information, budget)
        
        return {'success' : 'Your event extractor is being trained.'}
