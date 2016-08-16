
from random import sample, shuffle
import pickle
import sys
from app import app
from ml.extractors.cnn_core.test import test_cnn
from util import write_model_to_file, retrain
from crowdjs_util import make_labeling_crowdjs_task, make_recall_crowdjs_task, make_precision_crowdjs_task
import urllib2
from schema.job import Job
from math import floor, ceil


def test_controller(task_information, task_category_id):


    some_examples_to_test_with = []
    with open('data/test_data/general_events_death', 'r') as f:
        for example in f:
            some_examples_to_test_with.append(example)
            
    if task_category_id == 2:
        task = make_labeling_crowdjs_task(some_examples_to_test_with,
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
def greedy_controller(task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Greedy Controller activated."
    sys.stdout.flush()
    
    (event_name, event_definition,
     event_good_example_1,
     event_good_example_1_trigger,
     event_good_example_2,
     event_good_example_2_trigger,
     event_bad_example_1,
     event_negative_good_example_1,
     event_negative_bad_example_1) = task_information
    
    if costSoFar >= (budget / 2):
        print "choosing to find examples from TACKBP and label them"
        sys.stdout.flush()
        next_category = app.config['EXAMPLE_CATEGORIES'][2]

        budget_left_over = budget - costSoFar
        cost_of_one_example = app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
        budget_left_over = int(ceil(budget_left_over / cost_of_one_example))
        num_positive_examples_to_label = int(budget_left_over / 2)
        num_negative_examples_to_label = (budget_left_over -
                                          num_positive_examples_to_label)

        print "Numbers:"
        print budget_left_over
        print num_positive_examples_to_label
        print num_negative_examples_to_label
        sys.stdout.flush()

        retrain(job_id, ['all'])

        test_examples = []
        test_labels = []

        tackbp_newswire_corpus = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/tac-kbp-2009/SENTTEXTINFORMATION_UNNUMBERED_270k')
        for sentence in tackbp_newswire_corpus:
            test_examples.append(sentence)
            test_labels.append(0)
            
        job = Job.objects.get(id = job_id)
        vocabulary = pickle.loads(job.vocabulary)

        predicted_labels = test_cnn(
            test_examples,
            test_labels,
            write_model_to_file(job_id),
            vocabulary)

        positive_examples = []
        negative_examples = []
        for i in range(len(predicted_labels)):
            predicted_label = predicted_labels[i]
            example = test_examples[i]
            if predicted_label == 1:
                positive_examples.append(example)
            else:
                negative_examples.append(example)

        print "Sampling examples from the corpus"
        sys.stdout.flush()

        selected_examples = []
        if positive_examples < num_positive_examples_to_label:
            selected_examples += positive_examples
            selected_examples += sample(
                negative_examples,
                budget_left_over - len(positive_examples))
        elif negative_examples < num_negative_examples_to_label:
            selected_examples += negative_examples
            selected_examples += sample(
                positive_examples,
                budget_left_over- len(negative_examples))
        else:
            selected_examples += sample(positive_examples,
                                        num_positive_examples_to_label)
            selected_examples += sample(negative_examples,
                                        num_negative_examples_to_label)

        print "Shuffling examples from the corpus"
        sys.stdout.flush()

        shuffle(selected_examples)

        task = make_labeling_crowdjs_task(selected_examples, task_information)

        print "Number of hits:"
        print len(selected_examples)
        print app.config['CONTROLLER_LABELS_PER_QUESTION']
        print next_category['price']
        sys.stdout.flush()
        print len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION']


        print "Price:"
        len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
        sys.stdout.flush()
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % 2 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']

    #If task_categories has one element in it, pick a category that
    #can use previous training data
    if len(task_categories) % 2 == 1:

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        return next_category['id'], task, len(last_batch), len(last_batch)*next_category['price']
