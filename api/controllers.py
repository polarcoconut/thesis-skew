
from random import sample, shuffle, random
import pickle
import sys
from app import app
#from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores

from util import write_model_to_file, retrain, get_unlabeled_examples_from_corpus, get_random_unlabeled_examples_from_corpus, split_examples, test, get_US_unlabeled_examples_from_corpus, get_US_PP_unlabeled_examples_from_corpus
from crowdjs_util import make_labeling_crowdjs_task, make_recall_crowdjs_task, make_precision_crowdjs_task
import urllib2
from schema.job import Job
from schema.experiment import Experiment
from schema.gold_extractor import Gold_Extractor
from math import floor, ceil, sqrt, log
import numpy as np

import cPickle
from ml.extractors.cnn_core.test import test_cnn
import os
from sklearn.model_selection import KFold
from scipy import stats



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
        return 0, task, app.config['CONTROLLER_GENERATE_BATCH_SIZE'], 0

    elif task_category_id == 1:
        task = make_precision_crowdjs_task(some_examples_to_test_with,
                                        task_information)
        return 1, task, len(some_examples_to_test_with), 0


def label_only_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    next_category = app.config['EXAMPLE_CATEGORIES'][2]
    
    (selected_examples, 
     expected_labels) = get_random_unlabeled_examples_from_corpus(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id)
    
    task = make_labeling_crowdjs_task(selected_examples,
                                      expected_labels,
                                      task_information)
    
    return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']





def label_only_US_controller(
        task_ids, task_categories, training_examples,
        training_labels, task_information,
        costSoFar, budget, job_id):
    

    next_category = app.config['EXAMPLE_CATEGORIES'][2]


    if len(task_ids) == 0:
        
        (selected_examples, 
         expected_labels) = get_random_unlabeled_examples_from_corpus(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
    else:
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    
def round_robin_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Round-Robin Controller activated."
    sys.stdout.flush()
        
    if len(task_categories) % 3 == 2:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_corpus(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % 3 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']

    if len(task_categories) % 3 == 1:

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']

def round_robin_no_negate_controller(task_ids, task_categories, 
                                     training_examples,
                                     training_labels, task_information,
                                     costSoFar, budget, job_id):


    print "Round-Robin No-Negate Controller activated."
    sys.stdout.flush()
        
    if len(task_categories) % 2 == 1:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_corpus(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % 2 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']




#Pick the action corresponding to the distribution that the extractor performs #most poorly on.
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
        predicted_labels = test(
            job_id,
            validation_all_examples,
            validation_all_labels)

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
         expected_labels) = get_unlabeled_examples_from_corpus(
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
        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return 0, task, num_hits, num_hits * next_category['price']
    
    elif worst_task_category_id == 1:
        print "choosing the PRECISION category"
        sys.stdout.flush()

        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        #positive_examples = []

        generate_task_ids = categories_to_examples[0]
        positive_examples, negative_examples = split_examples(
                generate_task_ids,
                [0 for i in generate_task_ids],
                ['all'])
        #for training_example_set, training_label_set in zip(
        #        training_examples, training_labels):
        #    for training_example, training_label in zip(
        #            training_example_set, training_label_set):
        #        if training_label == 1:
        #            positive_examples.append(training_example)

        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']

        selected_positive_examples = sample(positive_examples, num_hits)
        
        
        task = make_precision_crowdjs_task(selected_positive_examples,
                                           task_information)
        
        return 1, task, num_hits, num_hits * next_category['price']




#Alternate back and forth between precision and recall categories.
#Then, use the other half of the budget and
#select a bunch of examples from corpus to label.
#THIS CONTROLLER CAN ONLY BE USED DURING EXPERIMENTS BECAUSE IT REQUIRES
#GOLD LABELS
def seed_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Seed Controller activated."
    sys.stdout.flush()

    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']
        
    task_categories_per_cycle = num_negatives_wanted + 1
        
    if len(task_categories) >= (task_categories_per_cycle * 
                                num_negatives_wanted):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_corpus_at_fixed_ratio(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % task_categories_per_cycle == 0:

        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] *
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']

#Alternate back and forth between precision and recall categories.
#Then, use the other half of the budget and
#select a bunch of examples from corpus to label.
#THIS CONTROLLER CAN ONLY BE USED DURING EXPERIMENTS BECAUSE IT REQUIRES
#GOLD LABELS
def seed_US_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Seed Controller activated."
    sys.stdout.flush()

    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']
        
    task_categories_per_cycle = num_negatives_wanted + 1
        
    if len(task_categories) >= (task_categories_per_cycle * 
                                num_negatives_wanted):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % task_categories_per_cycle == 0:

        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] *
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']








def round_robin_US_controller(task_ids, task_categories, 
                              training_examples,
                              training_labels, task_information,
                              costSoFar, budget, job_id):
    

    print "RR US Controller activated."
    sys.stdout.flush()
        
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 2

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):
        print "choosing the PRECISION category"
        sys.stdout.flush()

        
        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        

        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if (len(task_categories) % task_categories_per_cycle  == 
        num_negatives_wanted + 1):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        print "choosing the LABEL category"
        sys.stdout.flush()


            
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
            
    

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']



def ucb_controller(task_ids, task_categories, 
                   training_examples,
                   training_labels, task_information,
                   costSoFar,
                   extra_job_state,
                   budget, job_id):
    
    print "UCB Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_mean_costs'] = {
            0 : app.config['EXAMPLE_CATEGORIES'][0]['price'], 
            1 : 0, 
            2: 0}
    else:

        last_action = task_categories[-1]
        empirical_skew = 0.0
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:
            for label in training_labels[-1]:
                if label == 1:
                    empirical_skew += 1
            #empirical_skew /= len(training_labels[-1])

            if empirical_skew == 0:
                empirical_skew = app.config['UCB_SMOOTHING_PARAMETER']

            empirical_cost_of_positive = (
                app.config['EXAMPLE_CATEGORIES'][2]['price']  *
                app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 
                empirical_skew)
                
            print "UPDATING THE EMPIRICAL SKEW"
            print empirical_skew
            print app.config['EXAMPLE_CATEGORIES'][2]['price']
            print  app.config['CONTROLLER_LABELING_BATCH_SIZE']
                                                      
            sys.stdout.flush()
            
            old_mean_cost = extra_job_state['action_mean_costs'][2]
            extra_job_state['action_mean_costs'][2] = (
                ((extra_job_state['action_counts'][2] - 1) * old_mean_cost +
                 empirical_cost_of_positive) / 
                extra_job_state['action_counts'][2])            
           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute the upper confidence bounds

    selected_action = None
    
    if extra_job_state['action_counts'][2] == 0:
        selected_action = 2
    else:

        num_batches = len(training_examples)
        
        cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']
        #cost_of_action_0 = (extra_job_state['action_mean_costs'][0] - 
        #                    sqrt(2.0 * log(num_batches) / 
        #                         extra_job_state['action_counts'][0]))
        c = app.config['UCB_EXPLORATION_CONSTANT']
        cost_of_action_2 = (extra_job_state['action_mean_costs'][2] - 
                            sqrt(c * log(num_batches) / 
                                 extra_job_state['action_counts'][2]))
        
        print "COSTS OF ACTIONS"
        print cost_of_action_0
        print extra_job_state['action_mean_costs'][0]
        print num_batches
        print extra_job_state['action_counts'][0]
        print "-----"
        print cost_of_action_2
        print extra_job_state['action_mean_costs'][2]
        print num_batches
        print extra_job_state['action_counts'][2]

        sys.stdout.flush()


        if cost_of_action_0 < cost_of_action_2:
            selected_action = 0
        else:
            selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()
            
            (selected_examples, 
             expected_labels) = get_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)

            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

def ucb_US_controller(task_ids, task_categories, 
                   training_examples,
                   training_labels, task_information,
                   costSoFar,
                   extra_job_state,
                   budget, job_id):
    
    print "UCB with Active Learning Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_mean_costs'] = {
            0 : app.config['EXAMPLE_CATEGORIES'][0]['price'], 
            1 : 0, 
            2: 0}
    else:

        last_action = task_categories[-1]
        empirical_skew = 0.0
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:
            for label in training_labels[-1]:
                if label == 1:
                    empirical_skew += 1
            #empirical_skew /= len(training_labels[-1])

            if empirical_skew == 0:
                empirical_skew = app.config['UCB_SMOOTHING_PARAMETER']

            empirical_cost_of_positive = (
                app.config['EXAMPLE_CATEGORIES'][2]['price']  *
                app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 
                empirical_skew)
                
            print "UPDATING THE EMPIRICAL SKEW"
            print empirical_skew
            print app.config['EXAMPLE_CATEGORIES'][2]['price']
            print  app.config['CONTROLLER_LABELING_BATCH_SIZE']
                                                      
            sys.stdout.flush()
            
            old_mean_cost = extra_job_state['action_mean_costs'][2]
            extra_job_state['action_mean_costs'][2] = (
                ((extra_job_state['action_counts'][2] - 1) * old_mean_cost +
                 empirical_cost_of_positive) / 
                extra_job_state['action_counts'][2])            
           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute the upper confidence bounds

    selected_action = None
    
    if extra_job_state['action_counts'][2] == 0:
        selected_action = 2
    else:

        num_batches = len(training_examples)
        
        cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']
        #cost_of_action_0 = (extra_job_state['action_mean_costs'][0] - 
        #                    sqrt(2.0 * log(num_batches) / 
        #                         extra_job_state['action_counts'][0]))
        c = app.config['UCB_EXPLORATION_CONSTANT']
        cost_of_action_2 = (extra_job_state['action_mean_costs'][2] - 
                            sqrt(c * log(num_batches) / 
                                 extra_job_state['action_counts'][2]))
        
        print "COSTS OF ACTIONS"
        print cost_of_action_0
        print extra_job_state['action_mean_costs'][0]
        print num_batches
        print extra_job_state['action_counts'][0]
        print "-----"
        print cost_of_action_2
        print extra_job_state['action_mean_costs'][2]
        print num_batches
        print extra_job_state['action_counts'][2]

        sys.stdout.flush()


        if cost_of_action_0 < cost_of_action_2:
            selected_action = 0
        else:
            selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()

            (selected_examples, 
             expected_labels) = get_US_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']



def ucb_US_PP_controller(task_ids, task_categories, 
                         training_examples,
                         training_labels, task_information,
                         costSoFar,
                         extra_job_state,
                         budget, job_id):
    
    print "UCB with Active Learning Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_mean_costs'] = {
            0 : app.config['EXAMPLE_CATEGORIES'][0]['price'], 
            1 : 0, 
            2: 0}
    else:

        last_action = task_categories[-1]
        empirical_skew = 0.0
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:
            for label in training_labels[-1]:
                if label == 1:
                    empirical_skew += 1
            #empirical_skew /= len(training_labels[-1])

            if empirical_skew == 0:
                empirical_skew = app.config['UCB_SMOOTHING_PARAMETER']

            empirical_cost_of_positive = (
                app.config['EXAMPLE_CATEGORIES'][2]['price']  *
                app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 
                empirical_skew)
                
            print "UPDATING THE EMPIRICAL SKEW"
            print empirical_skew
            print app.config['EXAMPLE_CATEGORIES'][2]['price']
            print  app.config['CONTROLLER_LABELING_BATCH_SIZE']
                                                      
            sys.stdout.flush()
            
            old_mean_cost = extra_job_state['action_mean_costs'][2]
            extra_job_state['action_mean_costs'][2] = (
                ((extra_job_state['action_counts'][2] - 1) * old_mean_cost +
                 empirical_cost_of_positive) / 
                extra_job_state['action_counts'][2])            
           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute the upper confidence bounds

    selected_action = None
    
    if extra_job_state['action_counts'][2] == 0:
        selected_action = 2
    else:

        num_batches = len(training_examples)
        
        cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']
        #cost_of_action_0 = (extra_job_state['action_mean_costs'][0] - 
        #                    sqrt(2.0 * log(num_batches) / 
        #                         extra_job_state['action_counts'][0]))
        c = app.config['UCB_EXPLORATION_CONSTANT']
        cost_of_action_2 = (extra_job_state['action_mean_costs'][2] - 
                            sqrt(c * log(num_batches) / 
                                 extra_job_state['action_counts'][2]))
        
        print "COSTS OF ACTIONS"
        print cost_of_action_0
        print extra_job_state['action_mean_costs'][0]
        print num_batches
        print extra_job_state['action_counts'][0]
        print "-----"
        print cost_of_action_2
        print extra_job_state['action_mean_costs'][2]
        print num_batches
        print extra_job_state['action_counts'][2]

        sys.stdout.flush()


        if cost_of_action_0 < cost_of_action_2:
            selected_action = 0
        else:
            selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()

            (selected_examples, 
             expected_labels) = get_US_PP_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']



def thompson_controller(task_ids, task_categories, 
                   training_examples,
                   training_labels, task_information,
                   costSoFar,
                   extra_job_state,
                   budget, job_id):
    
    print "Thompson Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_beta'] = {
            0 : None, 
            1 : None, 
            2 : [5,1]}
    else:
        last_action = task_categories[-1]
        
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:

            num_positives_retrieved = 0
            num_negatives_retrieved = 0

            for label in training_labels[-1]:
                if label == 1:
                    num_positives_retrieved += 1
                elif label == 0:
                    num_negatives_retrieved += 1
                else:
                    raise Exception

            extra_job_state['action_beta'][2][0] += num_positives_retrieved
            extra_job_state['action_beta'][2][1] += num_negatives_retrieved
            
            print "UPDATING THE BETA DISTRIBUTION"
            print extra_job_state['action_beta'][2]
            sys.stdout.flush()

           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute a distribution on the upper confidence bounds

    selected_action = None
    
    num_batches = len(training_examples)

    cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']

    #Draw a skew sample
    skew_sample = np.random.beta(
        extra_job_state['action_beta'][2][0],
        extra_job_state['action_beta'][2][1])
    num_of_positives_sample = skew_sample * app.config[
        'CONTROLLER_LABELING_BATCH_SIZE']

    #based on skew sample, compute cost.
    cost_of_action_2 = (
        app.config['EXAMPLE_CATEGORIES'][2]['price']  *
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] /
        num_of_positives_sample)

    print "COSTS OF ACTIONS"
    print cost_of_action_0
    print skew_sample
    print num_of_positives_sample
    print cost_of_action_2
    print extra_job_state['action_beta'][2]
    sys.stdout.flush()


    if cost_of_action_0 < cost_of_action_2:
        selected_action = 0
    else:
        selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()
            
            (selected_examples, 
             expected_labels) = get_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)

            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']


def thompson_US_controller(task_ids, task_categories, 
                   training_examples,
                   training_labels, task_information,
                   costSoFar,
                   extra_job_state,
                   budget, job_id):
    
    print "Thompson US Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_beta'] = {
            0 : None, 
            1 : None, 
            2 : [5,1]}
    else:
        last_action = task_categories[-1]
        
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:

            num_positives_retrieved = 0
            num_negatives_retrieved = 0

            for label in training_labels[-1]:
                if label == 1:
                    num_positives_retrieved += 1
                elif label == 0:
                    num_negatives_retrieved += 1
                else:
                    raise Exception

            extra_job_state['action_beta'][2][0] += num_positives_retrieved
            extra_job_state['action_beta'][2][1] += num_negatives_retrieved
            
            print "UPDATING THE BETA DISTRIBUTION"
            print extra_job_state['action_beta'][2]
            sys.stdout.flush()

           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute a distribution on the upper confidence bounds

    selected_action = None
    
    num_batches = len(training_examples)

    cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']

    #Draw a skew sample
    skew_sample = np.random.beta(
        extra_job_state['action_beta'][2][0],
        extra_job_state['action_beta'][2][1])
    num_of_positives_sample = skew_sample * app.config[
        'CONTROLLER_LABELING_BATCH_SIZE']

    #based on skew sample, compute cost.
    cost_of_action_2 = (
        app.config['EXAMPLE_CATEGORIES'][2]['price']  *
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] /
        num_of_positives_sample)

    print "COSTS OF ACTIONS"
    print cost_of_action_0
    print skew_sample
    print num_of_positives_sample
    print cost_of_action_2
    print extra_job_state['action_beta'][2]
    sys.stdout.flush()


    if cost_of_action_0 < cost_of_action_2:
        selected_action = 0
    else:
        selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()
            
            (selected_examples, 
             expected_labels) = get_US_unlabeled_examples_from_corpus(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)

            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']



def hybrid_controller(task_ids, task_categories, 
                      training_examples,
                      training_labels, task_information,
                      costSoFar,
                      extra_job_state,
                      budget, job_id,
                      threshold):
    

    print "Hybrid activated."
    sys.stdout.flush()


    if not extra_job_state:
        extra_job_state['estimated_f1s'] = []
        extra_job_state['costSoFars'] = []

        print "choosing the RECALL category"
        sys.stdout.flush()
        
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
               
        task = make_recall_crowdjs_task(task_information)
               
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']

    last_action = task_categories[-1]
    if last_action == 2:
        print "choosing the LABEL category"
        sys.stdout.flush()

        next_category = app.config['EXAMPLE_CATEGORIES'][2]

        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)

        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']


    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 1

    #Count the number of times we've called the precision category
    number_of_precision_actions = 0
    i = -1
    while task_categories[i] == 1:
        number_of_precision_actions += 1
        i -= 1

    if number_of_precision_actions < num_negatives_wanted:
        print "choosing the PRECISION category"
        sys.stdout.flush()

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, 
                                           task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])

        return next_category['id'], task, num_hits, num_hits*next_category['price']
    else:
        #determine if you've reached the threshold

        all_training_examples = []
        all_training_labels = []
        for training_example_set, training_label_set in zip(
                training_examples,training_labels):
            for training_example, training_label in zip(
                    training_example_set,training_label_set):
                all_training_examples.append(training_example)
                all_training_labels.append(training_label)


        average_f1 = 0.0

        kf = KFold(n_splits = 5)
        kf.get_n_splits(all_training_examples)

        for train_indices, test_indices in kf.split(all_training_examples):
            train_training_examples = [
                all_training_examples[i] for i in train_indices]
            train_training_labels = [
                all_training_labels[i] for i in train_indices]
            test_training_examples = [
                all_training_examples[i] for i in test_indices]
            test_training_labels = [
                all_training_labels[i] for i in test_indices]

            train_training_positive_examples = []
            train_training_negative_examples = []
            for ex, lab in zip(train_training_examples, 
                               train_training_labels):
                if lab == 1:
                    train_training_positive_examples.append(ex)
                elif lab == 0:
                    train_training_negative_examples.append(ex)
                else:
                    raise Exception

            retrain(job_id, ['all'], [],
                    train_training_positive_examples,
                    train_training_negative_examples)

            predicted_labels, label_probabilities = test(
                job_id,
                test_training_examples,
                test_training_labels)
            precision, recall, f1 = computeScores(predicted_labels, 
                                                  test_training_labels)
            average_f1 += f1
        average_f1 /= 5.0

        
        extra_job_state['costSoFars'].append(costSoFar)
        extra_job_state['estimated_f1s'].append(average_f1)

        print "ESTIMATED F1S"
        print extra_job_state['costSoFars']
        print extra_job_state['estimated_f1s']        
        sys.stdout.flush()


        if len(extra_job_state['estimated_f1s']) >= 6:
               
               slope, intercept, r_value, p_value, std_err = stats.linregress(
                   extra_job_state['costSoFars'][-4:],
                   extra_job_state['estimated_f1s'][-4:])
               

               print "CALCULATED SLOPE"
               print slope
               print intercept
               print threshold
               sys.stdout.flush()
        
               if slope < threshold:
                   print "choosing the LABEL category"
                   sys.stdout.flush()

                   next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
                   (selected_examples, 
                    expected_labels) = get_US_unlabeled_examples_from_corpus(
                        task_ids, task_categories,
                        training_examples, training_labels,
                        task_information, costSoFar,
                        budget, job_id)

                   task = make_labeling_crowdjs_task(selected_examples,
                                                     expected_labels,
                                                     task_information)

                   return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

        print "choosing the RECALL category"
        sys.stdout.flush()
               
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
               
        task = make_recall_crowdjs_task(task_information)
               
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


def guided_learning_controller(
        task_ids, task_categories, 
        training_examples,
        training_labels, task_information,
        costSoFar, budget, job_id):
    

    print "Guided Learning Controller activated."
    sys.stdout.flush()
        
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 1

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):
        print "choosing the PRECISION category"
        sys.stdout.flush()

        
        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        

        return next_category['id'], task, num_hits, num_hits*next_category['price']

