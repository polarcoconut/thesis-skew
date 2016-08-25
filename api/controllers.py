
from random import sample, shuffle
import pickle
import sys
from app import app
from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores

from util import write_model_to_file, retrain, get_unlabeled_examples_from_tackbp, split_examples
from crowdjs_util import make_labeling_crowdjs_task, make_recall_crowdjs_task, make_precision_crowdjs_task
import urllib2
from schema.job import Job
from math import floor, ceil


def test_controller(task_information, task_category_id):


    some_examples_to_test_with = []
    expected_labels = []
    with open('data/test_data/general_events_death', 'r') as f:
        for example in f:
            some_examples_to_test_with.append(example)
            expected_labels.append(-1)
            
    if task_category_id == 2:
        task = make_labeling_crowdjs_task(some_examples_to_test_with,
                                          expected_labels,
                                          task_information)
        return 2, task, len(some_examples_to_test_with) * app.config['CONTROLLER_LABELS_PER_QUESTION'], 0

    elif task_category_id == 0:
        task = make_recall_crowdjs_task(task_information)
        return 0, task, app.config['CONTROLLER_BATCH_SIZE'], 0

    elif task_category_id == 1:
        task = make_recall_crowdjs_task(some_examples_to_test_with,
                                        task_information)
        return 1, task, len(some_examples_to_test_with), 0

    
#Alternate back and forth between precision and recall categories.
#Then, use the other half of the budget and
#select a bunch of examples from TACKBP corpus to label.
def round_robin_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Round-Robin Controller activated."
    sys.stdout.flush()
        
    if len(task_categories) % 3 == 2:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        selected_examples, expected_labels = get_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % 3 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']

    #If task_categories has one element in it, pick a category that
    #can use previous training data
    if len(task_categories) % 3 == 1:

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        return next_category['id'], task, len(last_batch), len(last_batch)*next_category['price']


#Pick the action corresponding to the distributino that the extractor performs
#most poorly on.
def uncertainty_sampling_controller(task_ids, task_categories,
                                    training_examples,
                                     training_labels, task_information,
                                     costSoFar, budget, job_id):



    print "Uncertainty Sampling Controller activated."
    sys.stdout.flush()

    if len(task_categories) < 3:
        return  round_robin_controller(
            task_ids,task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    categories_to_examples = {}
    for i, task_category in zip(range(len(task_categories)), task_categories):

        #This check is because some data in the database is inconsistent
        if isinstance(task_category, dict):
            task_category_id = task_category['id']
        else:
            task_category_id = task_category

        if not task_category_id in categories_to_examples:
            categories_to_examples[task_category_id] = []

        categories_to_examples[task_category_id].append(task_ids[i])

    #For every kind of action, check to see how well the extractor can
    #predict it
    worst_task_category_id = []
    worst_fscore = 1.0
    for target_task_category_id in categories_to_examples.keys():

        training_positive_examples = []
        training_negative_examples = []
        validation_positive_examples = []
        validation_negative_examples = []
        validation_all_examples = []
        validation_all_labels = []
        
        for task_category_id  in categories_to_examples.keys():
            matching_task_ids = categories_to_examples[task_category_id]
            pos_examples, neg_examples = split_examples(
                matching_task_ids,
                [task_category_id for i in matching_task_ids],
                ['all'])
            if not task_category_id == target_task_category_id:
                training_positive_examples += pos_examples
                training_negative_examples += neg_examples
            else:
                shuffle(pos_examples)
                shuffle(neg_examples)

                size_of_validation_positive_examples = int(
                    ceil(0.2 * len(pos_examples)))
                size_of_validation_negative_examples = int(
                    ceil(0.2 * len(neg_examples)))
                
                validation_positive_examples += pos_examples[
                    0:size_of_validation_positive_examples]
                validation_negative_examples += neg_examples[
                    0:size_of_validation_negative_examples]

                training_positive_examples += pos_examples[
                    size_of_validation_positive_examples:]
                training_negative_examples += neg_examples[
                    size_of_validation_negative_examples:]

        validation_all_examples = (validation_positive_examples +
                                   validation_negative_examples)
        validation_all_labels = (
            [1 for e in range(len(validation_positive_examples))]+
            [0 for e in range(len(validation_negative_examples))])

        print "RETRAINING TO FIGURE OUT WHAT ACTION TO DO NEXT"
        print len(training_positive_examples)
        print len(training_negative_examples)
        print len(validation_all_examples)
        
        retrain(job_id, ['all'],
                training_positive_examples = training_positive_examples,
                training_negative_examples = training_negative_examples)

        job = Job.objects.get(id = job_id)
        vocabulary = pickle.loads(job.vocabulary)
        predicted_labels = test_cnn(
            validation_all_examples,
            validation_all_labels,
            write_model_to_file(job_id),
            vocabulary)

        precision, recall, f1 = computeScores(predicted_labels,
                                              validation_all_labels)

        print "Action:"
        print target_task_category_id
        print "Scores:"
        print precision, recall, f1
        sys.stdout.flush()

        if f1 < worst_fscore:
            worst_fscore =  f1
            worst_task_category_id = [target_task_category_id]
        elif f1 == worst_fscore:
            worst_task_category_id.append(target_task_category_id)
            
    print "Worst F Score"
    print worst_fscore
    sys.stdout.flush()

    worst_task_category_id = sample(worst_task_category_id, 1)[0]
    
    if worst_task_category_id == 2:
        print "choosing the LABEL category"
        sys.stdout.flush()

        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples,
         expected_labels) = get_unlabeled_examples_from_tackbp(
             task_ids, task_categories, training_examples,
             training_labels, task_information, costSoFar,
             budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
        
        return 2, task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    elif worst_task_category_id == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
        
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
        
        num_hits = app.config['CONTROLLER_BATCH_SIZE']
        return 0, task, num_hits, num_hits * next_category['price']
    
    elif worst_task_category_id == 1:
        print "choosing the PRECISION category"
        sys.stdout.flush()

        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        positive_examples = []
        
        for training_example_set, training_label_set in zip(
                training_examples, training_labels):
            for training_example, training_label in zip(
                    training_example_set, training_label_set):
                if training_label == 1:
                    positive_examples.append(training_example)

        num_hits = app.config['CONTROLLER_BATCH_SIZE']
        selected_positive_examples = sample(positive_examples, num_hits)
        
        
        task = make_precision_crowdjs_task(selected_positive_examples,
                                           task_information)
        
        return 1, task, num_hits, num_hits * next_category['price']

