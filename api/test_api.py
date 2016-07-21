from flask.ext.restful import reqparse, abort, Api, Resource
from flask import url_for, redirect, render_template
import json
import string
import pickle
from app import app
from train import getLatestCheckpoint, split_examples
import sys
import pickle
from ml.extractors.cnn_core.test import test_cnn
from ml.computeScores import computeScores
from util import write_model_to_file
from schema.job import Job

test_parser = reqparse.RequestParser()
test_parser.add_argument('job_id', type=str, required=True)
test_parser.add_argument('test_sentence', type=str, required=True)

cv_parser = reqparse.RequestParser()
cv_parser.add_argument('job_id', type=str, required=True)

class TestExtractorApi(Resource):
    def get(self):

        args = test_parser.parse_args()
        job_id = args['job_id']
        test_sentence = args['test_sentence']

        print "Extracting event from sentence:"
        print test_sentence
        sys.stdout.flush()

        job = Job.objects.get(id = job_id)
        vocabulary = pickle.loads(job.vocabulary)
        predicted_labels = test_cnn([test_sentence], [0],
                                    write_model_to_file(job_id),
                                    vocabulary)

        print "predicted_labels"
        print predicted_labels
        sys.stdout.flush()
            
        return predicted_labels[0]

class CrossValidationExtractorApi(Resource):
    def get(self):

        args = cv_parser.parse_args()
        job_id = args['job_id']

        print "Doing cross validation"
        sys.stdout.flush()

        task_information, budget, checkpoint = getLatestCheckpoint(
            job_id, app.config)
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        test_positive_examples, test_negative_examples = split_examples(
            task_ids[-2:],
            task_categories[-2:],
            app.config)
        test_labels = ([1 for e in test_positive_examples] +
                       [0 for e in test_negative_examples])

        job = Job.objects.get(id = job_id)
        vocabulary = pickle.loads(job.vocabulary)

        predicted_labels = test_cnn(
            test_positive_examples + test_negative_examples,
            test_labels,
            write_model_to_file(job_id),
            vocabulary)

        print "predicted_labels"
        print predicted_labels
        sys.stdout.flush()

        precision, recall, f1 = computeScores(predicted_labels, test_labels)


        true_positives = []
        false_positives = []
        true_negatives = []
        false_negatives = []

        for example, label in zip(
                test_positive_examples,
                predicted_labels[0:len(test_positive_examples)]):
            if label == 1:
                true_positives.append(example)
            else:
                false_negatives.append(example)

        for example, label in zip(
                test_negative_examples,
                predicted_labels[len(test_positive_examples):]):
            if label == 1:
                false_positives.append(example)
            else:
                true_negatives.append(example)


        return (true_positives,
                false_positives,
                true_negatives,
                false_negatives,
                [precision, recall, f1])
