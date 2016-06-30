from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import train
import sys
import pickle
from ml.extractors.cnn_core.test import test_cnn
import redis

test_parser = reqparse.RequestParser()

test_parser.add_argument('job_id', type=str, required=True)
test_parser.add_argument('test_sentence', type=str, required=True)

class TestExtractorApi(Resource):
    def get(self):

        args = test_parser.parse_args()
        job_id = args['job_id']
        test_sentence = args['test_sentence']

        print "Extracting event from sentence:"
        print test_sentence
        sys.stdout.flush()

        model_file_name = app.redis.hmget(job_id, 'model_file_name')[0]
        model = app.redis.hmget(job_id,'model')[0]
                                
        model_file_handle = open(model_file_name, 'wb')
        model_file_handle.write(model)
        model_file_handle.close()

        vocabulary = pickle.loads(app.redis.hmget(job_id, 'vocabulary')[0])
        predicted_labels = test_cnn([test_sentence], [0], model_file_name,
                                    vocabulary)

        print "predicted_labels"
        print predicted_labels
        sys.stdout.flush()
            
        return predicted_labels[0]
