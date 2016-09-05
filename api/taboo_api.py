from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for
import json
import string
import pickle
from app import app
from util import compute_taboo_words
import sys

taboo_parser = reqparse.RequestParser()

#taboo_parser.add_argument('old_taboo_words', type=str, required=True)
taboo_parser.add_argument('old_sentence', type=str, required=True)
taboo_parser.add_argument('new_sentence', type=str, required=True)
taboo_parser.add_argument('task_id', type=str, required=True)
taboo_parser.add_argument('requester_id', type=str, required=True)

class ComputeTabooApi(Resource):
    def post(self):
        args = taboo_parser.parse_args()
        #old_taboo_words = args['old_taboo_words']
        old_sentence = args['old_sentence']
        new_sentence = args['new_sentence']
        task_id = args['task_id']
        requester_id = args['requester_id']
        
        print "Posting to"
        print app.config['CROWDJS_PUT_TASK_DATA_URL']
        sys.stdout.flush()

        compute_taboo_words.delay(old_sentence,
                                  new_sentence,
                                  task_id,
                                  requester_id,
                                  app.config['CROWDJS_PUT_TASK_DATA_URL'])
        
        return {'success' : 'New taboo words will be computed'}

