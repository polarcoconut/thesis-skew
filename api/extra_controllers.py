
from random import sample, shuffle, random
import pickle
import sys
from app import app
#from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores

from util import write_model_to_file, retrain, get_unlabeled_examples_from_corpus, get_random_unlabeled_examples_from_corpus, split_examples
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

#Pick the action corresponding to the distribution that the extractor performs
#most poorly on.
def impact_sampling_controller(task_ids, task_categories,
                               training_examples,
                               training_labels, task_information,
                               costSoFar, budget, job_id):
    


    print "Impact Sampling Controller activated."
    sys.stdout.flush()

    if len(task_categories) < 6:
        return  round_robin_controller(
            task_ids,task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    elif len(task_categories) < 12:
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
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    #First update the statistics about metric improvements from the last 
    #action taken

    last_task_id = task_ids[-1]
    last_task_category = task_categories[-1]

    categories_to_examples = {}
    for i, task_category in zip(range(len(task_categories)-1), 
                                task_categories[0:-1]):

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

    #Take the examples from the GENERATE category and use them to compute
    #recall.
    #Take the examples from the LABEL category and use them to compute 
    #precision.

    training_positive_examples = []
    training_negative_examples = []
    validation_recall_examples = []
    validation_recall_labels = []
    validation_precision_examples = []
    validation_precision_labels = []

    recall_measuring_task_cat_ids = [0]
    precision_measuring_task_cat_ids = [2]
    other_task_cat_ids = [1]

    for recall_measuring_task_cat_id in recall_measuring_task_cat_ids:
        recall_task_ids = categories_to_examples[
            recall_measuring_task_cat_id]
        
        recall_examples, placeholder = split_examples(
            recall_task_ids,
            [recall_measuring_task_cat_id for i in recall_task_ids],
            ['all'])
        
        if len(placeholder) > 0:
            raise Exception
            
        shuffle(recall_examples)

        size_of_validation_recall_examples = int(
            ceil(0.4 * len(recall_examples)))

        validation_recall_examples += recall_examples[
            0:size_of_validation_recall_examples]

        validation_recall_labels += [1 for e in range(
            size_of_validation_recall_examples)]

        training_positive_examples += recall_examples[
            size_of_validation_recall_examples:]
    

        print "ADDING RECALL EXAMPLES"
        print len(training_positive_examples)
        print len(training_negative_examples)
        sys.stdout.flush()

    for precision_measuring_task_cat_id in precision_measuring_task_cat_ids:
        precision_task_ids = categories_to_examples[
            precision_measuring_task_cat_id]

        pos_examples, neg_examples = split_examples(
            precision_task_ids,
            [precision_measuring_task_cat_id for i in precision_task_ids],
            ['all'])
    
        if len(placeholder) > 0:
            raise Exception
                        
        
        shuffled_indices = np.random.permutation(
            np.arange(len(pos_examples) + len(neg_examples)))

        size_of_validation_precision_examples = int(
            ceil(0.4 * len(shuffled_indices)))

        for index in shuffled_indices[0:size_of_validation_precision_examples]:
            if index < len(pos_examples):
                validation_precision_examples.append(pos_examples[index])
                validation_precision_labels.append(1)
            else:
                real_index = index - len(pos_examples)
                validation_precision_examples.append(neg_examples[real_index])
                validation_precision_labels.append(0)

        for index in shuffled_indices[size_of_validation_precision_examples:]:
            if index < len(pos_examples):
                training_positive_examples.append(pos_examples[index])
            else:
                real_index = index - len(pos_examples)
                training_negative_examples.append(neg_examples[real_index])

        print "ADDING PRECISION EXAMPLES"
        print len(training_positive_examples)
        print len(training_negative_examples)
        sys.stdout.flush()

        

    for other_task_cat_id in other_task_cat_ids:
       other_task_ids = categories_to_examples[other_task_cat_id]
       
       pos_examples, neg_examples = split_examples(
           other_task_ids,
           [other_task_cat_id for i in other_task_ids],
           ['all'])


       training_positive_examples += pos_examples
       training_negative_examples += neg_examples


       print "ADDING ALL OTHER EXAMPLES"
       print len(training_positive_examples)
       print len(training_negative_examples)
       sys.stdout.flush()
       

               
    print "RETRAINING TO FIGURE OUT WHAT ACTION TO DO NEXT"
    print len(training_positive_examples)
    print len(training_negative_examples)
    sys.stdout.flush()

    
    
    f1s = []

    for i in range(3):
        retrain(job_id, ['all'],
                training_positive_examples = training_positive_examples,
                training_negative_examples = training_negative_examples)
        
        job = Job.objects.get(id = job_id)
    
        predicted_labels = test(
            job_id,
            validation_recall_examples + validation_precision_examples,
            validation_recall_labels + validation_precision_labels)

        predicted_labels_for_recall_examples = predicted_labels[
            0:len(validation_recall_examples)]
        predicted_labels_for_precision_examples = predicted_labels[
            len(validation_recall_examples):]

        #compute scores separately for precision and recall
        _, recall, _ = computeScores(
            predicted_labels_for_recall_examples,
            validation_recall_labels)
        
        precision, _, _ = computeScores(
            predicted_labels_for_precision_examples,
            validation_precision_labels)
    
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        print recall
        print predicted_labels_for_recall_examples
        print validation_recall_labels
        print precision
        print predicted_labels_for_precision_examples
        print validation_precision_labels
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        sys.stdout.flush()
        
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (precision * recall) / (precision + recall)
            
        f1s.append(f1)
    f1 = np.mean(f1s)
    ## Add in the extra data and compute the effect
    
    print "ADDING BACK IN EXTRA DATA"
    print last_task_id
    print last_task_category
    sys.stdout.flush()
    
    pos_examples, neg_examples = split_examples(
        [last_task_id], [last_task_category], ['all'])
    
    
    training_positive_examples += pos_examples
    training_negative_examples += neg_examples
    

    
    new_f1s = []
    new_precisions = []
    new_recalls = []

    for i in range(3):
        retrain(job_id, ['all'],
                training_positive_examples = training_positive_examples,
                training_negative_examples = training_negative_examples)
    
        job = Job.objects.get(id = job_id)
        
        predicted_labels = test(
            job_id,
            validation_recall_examples + validation_precision_examples,
            validation_recall_labels + validation_precision_labels)

        predicted_labels_for_recall_examples = predicted_labels[
            0:len(validation_recall_examples)]
        predicted_labels_for_precision_examples = predicted_labels[
            len(validation_recall_examples):]
        
        
        #compute scores separately for precision and recall
        _, new_recall, _ = computeScores(
            predicted_labels_for_recall_examples,
            validation_recall_labels)
        
        new_precision, _, _ = computeScores(
            predicted_labels_for_precision_examples,
            validation_precision_labels)
        
        
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        print new_recall
        print predicted_labels_for_recall_examples
        print validation_recall_labels
        print new_precision
        print predicted_labels_for_precision_examples
        print validation_precision_labels
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        sys.stdout.flush()
        
        if (new_precision + new_recall) == 0:
            new_f1 = 0.0
        else:
            new_f1 = (2.0 * (new_precision * new_recall) / 
                      (new_precision + new_recall))
        new_f1s.append(new_f1)
        new_precisions.append(new_precision)
        new_recalls.append(new_recall)


    new_f1 = np.mean(new_f1s)
    new_precision = np.mean(new_precisions)
    new_recall = np.mean(new_recalls)
    
    change_in_f1 = new_f1 - f1


    current_control_data = pickle.loads(job.control_data)

    current_control_data[last_task_category].append(change_in_f1)
            
    job.control_data = pickle.dumps(current_control_data)
    job.save()

    print "------------------------------------------"
    print "------------------------------------------"
    print "------------------------------------------"
    print current_control_data
    print "------------------------------------------"
    print "------------------------------------------"
    print "------------------------------------------"
    sys.stdout.flush()


    if len(task_categories) < 15:
        return  round_robin_controller(
            task_ids,task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)


    #Add an exploration term 

    best_task_category = []
    best_change = float('-inf')
    num_actions_taken_so_far = 0.0
    for task_category in current_control_data.keys():
        num_actions_taken_so_far += len(current_control_data[task_category])

    computed_values_of_each_action = []
    for task_category in current_control_data.keys():
        average_change = np.average(
            current_control_data[task_category],
            weights=range(1, len(current_control_data[task_category])+1))
        exploration_term =  sqrt(
            2.0*log(num_actions_taken_so_far) / 
            len(current_control_data[task_category]) )
        c = app.config['UCB_EXPLORATION_CONSTANT']
        ucb_value = average_change + (c * exploration_term)

        computed_values_of_each_action.append(
            [current_control_data[task_category],
             average_change,
             exploration_term,
             ucb_value])
        
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        print "Value of action %d" % task_category
        print current_control_data[task_category]
        print average_change
        print exploration_term
        print ucb_value
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        sys.stdout.flush()

        if ucb_value > best_change:
            best_task_category = [task_category]
            best_change = ucb_value
        elif ucb_value == best_change:
            best_task_category.append(task_category)
    
    current_logging_data = pickle.loads(job.logging_data)
    current_logging_data.append([best_task_category,
                                 [new_precision, new_recall, new_f1],
                                 computed_values_of_each_action])
    job.logging_data = pickle.dumps(current_logging_data)
    job.save()

    #epsilon = 1.0 / num_actions_taken_so_far
    
    #if random() < epsilon:
    #    other_choices = [0,1,2]
    #    for item in best_task_category:
    #        other_choices.remove(item)
    #    best_task_category = sample(other_choices, 1)[0]
    #else:
    best_task_category = sample(best_task_category,1)[0]

    if best_task_category == 2:
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

    elif best_task_category == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
        
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return 0, task, num_hits, num_hits * next_category['price']
    
    elif best_task_category == 1:
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

#Pick the action corresponding to the distribution that the extractor performs
#most poorly on.
def greedy_controller(task_ids, task_categories,
                               training_examples,
                               training_labels, task_information,
                               costSoFar, budget, job_id):
    


    print "Greedy Controller activated."
    sys.stdout.flush()

    if len(task_categories) < 6:
        return  round_robin_controller(
            task_ids,task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    elif len(task_categories) < 12:
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
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']


    #if len(task_categories) < 15:
    #    return  round_robin_controller(
    #        task_ids,task_categories, training_examples,
    #        training_labels, task_information,
    #        costSoFar, budget, job_id)


    categories_to_examples = {}
    for i, task_category in zip(range(len(task_categories)), 
                                task_categories):

        #This check is because some data in the database is inconsistent
        if isinstance(task_category, dict):
            task_category_id = task_category['id']
        else:
            task_category_id = task_category

        if not task_category_id in categories_to_examples:
            categories_to_examples[task_category_id] = []

        categories_to_examples[task_category_id].append(task_ids[i])


    training_positive_examples = []
    training_negative_examples = []
    validation_recall_examples = []
    validation_recall_labels = []
    validation_precision_examples = []
    validation_precision_labels = []

    recall_measuring_task_cat_ids = [0]
    precision_measuring_task_cat_ids = [2]
    other_task_cat_ids = [1]

    for recall_measuring_task_cat_id in recall_measuring_task_cat_ids:
        recall_task_ids = categories_to_examples[
            recall_measuring_task_cat_id]
        
        recall_examples, placeholder = split_examples(
            recall_task_ids,
            [recall_measuring_task_cat_id for i in recall_task_ids],
            ['all'])
        
        if len(placeholder) > 0:
            raise Exception
            
        shuffle(recall_examples)

        size_of_validation_recall_examples = int(
            ceil(0.4 * len(recall_examples)))

        validation_recall_examples += recall_examples[
            0:size_of_validation_recall_examples]

        validation_recall_labels += [1 for e in range(
            size_of_validation_recall_examples)]

        training_positive_examples += recall_examples[
            size_of_validation_recall_examples:]
    

        print "ADDING RECALL EXAMPLES"
        print len(training_positive_examples)
        print len(training_negative_examples)
        sys.stdout.flush()

    for precision_measuring_task_cat_id in precision_measuring_task_cat_ids:
        precision_task_ids = categories_to_examples[
            precision_measuring_task_cat_id]

        pos_examples, neg_examples = split_examples(
            precision_task_ids,
            [precision_measuring_task_cat_id for i in precision_task_ids],
            ['all'])
    
        if len(placeholder) > 0:
            raise Exception
                        
        
        shuffled_indices = np.random.permutation(
            np.arange(len(pos_examples) + len(neg_examples)))

        size_of_validation_precision_examples = int(
            ceil(0.4 * len(shuffled_indices)))

        for index in shuffled_indices[0:size_of_validation_precision_examples]:
            if index < len(pos_examples):
                validation_precision_examples.append(pos_examples[index])
                validation_precision_labels.append(1)
            else:
                real_index = index - len(pos_examples)
                validation_precision_examples.append(neg_examples[real_index])
                validation_precision_labels.append(0)

        for index in shuffled_indices[size_of_validation_precision_examples:]:
            if index < len(pos_examples):
                training_positive_examples.append(pos_examples[index])
            else:
                real_index = index - len(pos_examples)
                training_negative_examples.append(neg_examples[real_index])

        print "ADDING PRECISION EXAMPLES"
        print len(training_positive_examples)
        print len(training_negative_examples)
        sys.stdout.flush()

        

    for other_task_cat_id in other_task_cat_ids:
       other_task_ids = categories_to_examples[other_task_cat_id]
       
       pos_examples, neg_examples = split_examples(
           other_task_ids,
           [other_task_cat_id for i in other_task_ids],
           ['all'])


       training_positive_examples += pos_examples
       training_negative_examples += neg_examples


       print "ADDING ALL OTHER EXAMPLES"
       print len(training_positive_examples)
       print len(training_negative_examples)
       sys.stdout.flush()
       

               
    print "RETRAINING TO FIGURE OUT WHAT ACTION TO DO NEXT"
    print len(training_positive_examples)
    print len(training_negative_examples)
    sys.stdout.flush()


    new_f1s = []
    new_precisions = []
    new_recalls = []

    for i in range(3):
        retrain(job_id, ['all'],
                training_positive_examples = training_positive_examples,
                training_negative_examples = training_negative_examples)
    
        job = Job.objects.get(id = job_id)
        
        predicted_labels = test(
            job_id,
            validation_recall_examples + validation_precision_examples,
            validation_recall_labels + validation_precision_labels)

        predicted_labels_for_recall_examples = predicted_labels[
            0:len(validation_recall_examples)]
        predicted_labels_for_precision_examples = predicted_labels[
            len(validation_recall_examples):]
        
        
        #compute scores separately for precision and recall
        _, new_recall, _ = computeScores(
            predicted_labels_for_recall_examples,
            validation_recall_labels)
        
        new_precision, _, _ = computeScores(
            predicted_labels_for_precision_examples,
            validation_precision_labels)
        
        
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        print new_recall
        print predicted_labels_for_recall_examples
        print validation_recall_labels
        print new_precision
        print predicted_labels_for_precision_examples
        print validation_precision_labels
        print "------------------------------------------"
        print "------------------------------------------"
        print "------------------------------------------"
        sys.stdout.flush()
        
        if (new_precision + new_recall) == 0:
            new_f1 = 0.0
        else:
            new_f1 = (2.0 * (new_precision * new_recall) / 
                      (new_precision + new_recall))
        new_f1s.append(new_f1)
        new_precisions.append(new_precision)
        new_recalls.append(new_recall)


    new_f1 = np.mean(new_f1s)
    new_precision = np.mean(new_precisions)
    new_recall = np.mean(new_recalls)
    


    if new_precision < new_recall:
        best_task_category = [1,2]
    elif new_recall < new_precision:
        best_task_category = [0]
    else:
        best_task_category = [0,1,2]
    
    current_logging_data = pickle.loads(job.logging_data)
    current_logging_data.append([best_task_category,
                                 [new_precision, new_recall, new_f1]])
    job.logging_data = pickle.dumps(current_logging_data)
    job.save()

    best_task_category = sample(best_task_category,1)[0]

    if best_task_category == 2:
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

    elif best_task_category == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
        
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return 0, task, num_hits, num_hits * next_category['price']
    
    elif best_task_category == 1:
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


