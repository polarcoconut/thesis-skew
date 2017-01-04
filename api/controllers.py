
from random import sample, shuffle, random
import pickle
import sys
from app import app
#from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores

from util import write_model_to_file, retrain, get_unlabeled_examples_from_tackbp, get_random_unlabeled_examples_from_tackbp, split_examples, test
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
     expected_labels) = get_random_unlabeled_examples_from_tackbp(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id)
    
    task = make_labeling_crowdjs_task(selected_examples,
                                      expected_labels,
                                      task_information)
    
    return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

def label_only_constant_ratio_controller(task_ids, task_categories,
                                        training_examples,
                                        training_labels, task_information,
                                        costSoFar, budget, job_id):

    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    (selected_examples, 
     expected_labels) = get_random_unlabeled_examples_from_tackbp(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id)

    job = Job.objects.get(id = job_id)
    experiment = Experiment.objects.get(id=job.experiment_id)
    gold_extractor = Gold_Extractor.objects.get(
        name=experiment.gold_extractor)
    model_file_name = write_model_to_file(
        gold_extractor = gold_extractor.name) 
    vocabulary = cPickle.loads(str(gold_extractor.vocabulary))             
    predicted_labels, label_probabilities = test_cnn(selected_examples,
                                [0 for i in selected_examples], 
                                model_file_name, 
                                vocabulary)

    os.remove(os.path.join(
        os.getcwd(), model_file_name))
    os.remove(os.path.join(
        os.getcwd(),'%s.meta' % model_file_name))


    expected_positive_examples = []
    expected_negative_examples = []
    for predicted_label, selected_example in zip(predicted_labels,
                                             selected_examples):
        if predicted_label == 1:
            expected_positive_examples.append(selected_example)
        elif predicted_label == 0:
            expected_negative_examples.append(selected_example)
        else:
            raise Exception

    # For now, put in a ratio of 1 positive : 2 negatives

    selected_examples = []
    expected_labels = []
    num_negatives_wanted = 3
    for pos_example in expected_positive_examples:
        selected_examples.append(pos_example)
        expected_labels.append(1)

        temp_num_negatives_wanted = num_negatives_wanted
        while temp_num_negatives_wanted > 0:
            if len(expected_negative_examples) > 0:
                selected_examples.append(expected_negative_examples.pop())
                expected_labels.append(0)
                temp_num_negatives_wanted -= 1
            else:
                break

    if len(expected_positive_examples) == 0:
        selected_examples += sample(expected_negative_examples, 
                                    num_negatives_wanted)
        expected_labels += [0 for i in range(num_negatives_wanted)]

    task = make_labeling_crowdjs_task(selected_examples,
                                      expected_labels,
                                      task_information)

    return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']



def label_only_US_constant_ratio_controller(task_ids, task_categories,
                                            training_examples,
                                            training_labels, task_information,
                                            costSoFar, budget, job_id):
    
    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    active_learning_threshold = app.config['AL_THRESHOLD']
    #Figure out how many positive examples we currently have
    
    num_current_pos_examples, num_current_neg_examples = split_examples(
        task_ids, task_categories, ['all'])
    
    if num_current_pos_examples >= active_learning_threshold:
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

    elif num_current_pos_examples < active_learning_threshold:
        (selected_examples, 
         expected_labels) = get_random_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        job = Job.objects.get(id = job_id)
        experiment = Experiment.objects.get(id=job.experiment_id)
        gold_extractor = Gold_Extractor.objects.get(
            name=experiment.gold_extractor)
        model_file_name = write_model_to_file(
            gold_extractor = gold_extractor.name) 
        vocabulary = cPickle.loads(str(gold_extractor.vocabulary))             
        predicted_labels, label_probabilities = test_cnn(selected_examples,
                                    [0 for i in selected_examples], 
                                    model_file_name, 
                                    vocabulary)

        os.remove(os.path.join(
            os.getcwd(), model_file_name))
        os.remove(os.path.join(
            os.getcwd(),'%s.meta' % model_file_name))


        expected_positive_examples = []
        expected_negative_examples = []
        for predicted_label, selected_example in zip(predicted_labels,
                                                 selected_examples):
            if predicted_label == 1:
                expected_positive_examples.append(selected_example)
            elif predicted_label == 0:
                expected_negative_examples.append(selected_example)
            else:
                raise Exception

        # For now, put in a ratio of 1 positive : 2 negatives

        selected_examples = []
        expected_labels = []
        num_negatives_wanted = 3
        for pos_example in expected_positive_examples:
            selected_examples.append(pos_example)
            expected_labels.append(1)

            temp_num_negatives_wanted = num_negatives_wanted
            while temp_num_negatives_wanted > 0:
                if len(expected_negative_examples) > 0:
                    selected_examples.append(expected_negative_examples.pop())
                    expected_labels.append(0)
                    temp_num_negatives_wanted -= 1
                else:
                    break

        if len(expected_positive_examples) == 0:
            selected_examples += sample(expected_negative_examples, 
                                        num_negatives_wanted)
            expected_labels += [0 for i in range(num_negatives_wanted)]

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
        
        selected_examples, expected_labels = get_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % 2 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']




#Pick the action corresponding to the distribution that the extractor performs
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
         expected_labels) = get_unlabeled_examples_from_tackbp(
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
         expected_labels) = get_unlabeled_examples_from_tackbp(
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
         expected_labels) = get_unlabeled_examples_from_tackbp(
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
         expected_labels) = get_unlabeled_examples_from_tackbp(
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


#Alternate back and forth between precision and recall categories.
#Then, use the other half of the budget and
#select a bunch of examples from TACKBP corpus to label.
#THIS CONTROLLER CAN ONLY BE USED DURING EXPERIMENTS BECAUSE IT REQUIRES
#GOLD LABELS
def seed_controller(task_ids, task_categories, training_examples,
                      training_labels, task_information,
                      costSoFar, budget, job_id):


    print "Seed Controller activated."
    sys.stdout.flush()
        
    if len(task_categories) >= 10:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        job = Job.objects.get(id = job_id)
        experiment = Experiment.objects.get(id=job.experiment_id)
        gold_extractor = Gold_Extractor.objects.get(
            name=experiment.gold_extractor)
        model_file_name = write_model_to_file(
            gold_extractor = gold_extractor.name) 
        vocabulary = cPickle.loads(str(gold_extractor.vocabulary))             
        predicted_labels, label_probabilities = test_cnn(selected_examples,
                                    [0 for i in selected_examples], 
                                    model_file_name, 
                                    vocabulary)

        os.remove(os.path.join(
            os.getcwd(), model_file_name))
        os.remove(os.path.join(
            os.getcwd(),'%s.meta' % model_file_name))

 
        expected_positive_examples = []
        expected_negative_examples = []
        for predicted_label, selected_example in zip(predicted_labels,
                                                 selected_examples):
            if predicted_label == 1:
                expected_positive_examples.append(selected_example)
            elif predicted_label == 0:
                expected_negative_examples.append(selected_example)
            else:
                raise Exception
        
        # For now, put in a ratio of 1 positive : 2 negatives
        
        selected_examples = []
        expected_labels = []
        num_negatives_wanted = 3
        for pos_example in expected_positive_examples:
            selected_examples.append(pos_example)
            expected_labels.append(1)

            temp_num_negatives_wanted = num_negatives_wanted
            while temp_num_negatives_wanted > 0:
                if len(expected_negative_examples) > 0:
                    selected_examples.append(expected_negative_examples.pop())
                    expected_labels.append(0)
                    temp_num_negatives_wanted -= 1
                else:
                    break

        if len(expected_positive_examples) == 0:
            selected_examples += sample(expected_negative_examples, 
                                        num_negatives_wanted)
            expected_labels += [0 for i in range(num_negatives_wanted)]

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

    if len(task_categories) % 2 == 1:

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']



def round_robin_constant_ratio_controller(task_ids, task_categories, 
                                          training_examples,
                                          training_labels, task_information,
                                          costSoFar, budget, job_id):


    print "RRCR Controller activated."
    sys.stdout.flush()
        

    if len(task_categories) % 5 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']

    if len(task_categories) % 5 >= 1 and len(task_categories) % 5 <= 3:

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if len(task_categories) % 5  == 4:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_tackbp(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        job = Job.objects.get(id = job_id)
        experiment = Experiment.objects.get(id=job.experiment_id)
        gold_extractor = Gold_Extractor.objects.get(
            name=experiment.gold_extractor)
        model_file_name = write_model_to_file(
            gold_extractor = gold_extractor.name) 
        vocabulary = cPickle.loads(str(gold_extractor.vocabulary))             
        predicted_labels, label_probabilities = test_cnn(selected_examples,
                                    [0 for i in selected_examples], 
                                    model_file_name, 
                                    vocabulary)

        os.remove(os.path.join(
            os.getcwd(), model_file_name))
        os.remove(os.path.join(
            os.getcwd(),'%s.meta' % model_file_name))

 
        expected_positive_examples = []
        expected_negative_examples = []
        for predicted_label, selected_example in zip(predicted_labels,
                                                 selected_examples):
            if predicted_label == 1:
                expected_positive_examples.append(selected_example)
            elif predicted_label == 0:
                expected_negative_examples.append(selected_example)
            else:
                raise Exception
        
        # For now, put in a ratio of 1 positive : 2 negatives
        
        selected_examples = []
        expected_labels = []
        num_negatives_wanted = 3
        for pos_example in expected_positive_examples:
            selected_examples.append(pos_example)
            expected_labels.append(1)

            temp_num_negatives_wanted = num_negatives_wanted
            while temp_num_negatives_wanted > 0:
                if len(expected_negative_examples) > 0:
                    selected_examples.append(expected_negative_examples.pop())
                    expected_labels.append(0)
                    temp_num_negatives_wanted -= 1
                else:
                    break

        if len(expected_positive_examples) == 0:
            selected_examples += sample(expected_negative_examples, 
                                        num_negatives_wanted)
            expected_labels += [0 for i in range(num_negatives_wanted)]

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
