from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import train, restart
import sys

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

#train_parser.add_argument('task_information', type=str, required=True)
train_parser.add_argument('budget', type=str, required=True)
train_parser.add_argument('restart', type=bool, default=False)


class TrainExtractorApi(Resource):
    def post(self):
        args = train_parser.parse_args()
        event_name = args['event_name']
        event_definition = args['event_definition']
        event_pos_example_1 = args['event_pos_example_1']
        event_pos_example_1_trigger = args['event_pos_example_1_trigger']
        event_pos_example_2 = args['event_pos_example_2']
        event_pos_example_2_trigger = args['event_pos_example_2_trigger']
        event_pos_example_nearmiss = args['event_pos_example_nearmiss']
        
        event_neg_example = args['event_neg_example']
        event_neg_example_nearmiss = args['event_neg_example_nearmiss']


        task_information = (event_name, event_definition,
                            event_pos_example_1,
                            event_pos_example_1_trigger,
                            event_pos_example_2,
                            event_pos_example_2_trigger,
                            event_pos_example_nearmiss,
                            event_neg_example,
                            event_neg_example_nearmiss)
        
        #task_information = args['task_information']
        budget = int(args['budget'])
        restart = args['restart']

        if restart:
            restart.delay(task_information, budget, app.config)
        else:
            train.delay(task_information, budget, app.config)
            
        return redirect(url_for(
            'status',  
            event_name = event_name,
            event_definition = event_definition,
            event_pos_example_1 = event_pos_example_1,
            event_pos_example_1_trigger = event_pos_example_1_trigger,
            event_pos_example_2 = event_pos_example_2,
            event_pos_example_2_trigger = event_pos_example_2_trigger,
            event_pos_example_nearmiss = event_pos_example_nearmiss,
            event_neg_example = event_neg_example,
            event_neg_example_nearmiss = event_neg_example_nearmiss))



