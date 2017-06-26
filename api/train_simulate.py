import time
from app import app
from controllers import round_robin_controller, round_robin_no_negate_controller, uncertainty_sampling_controller, label_only_controller, seed_controller, seed_US_controller, label_only_US_controller, ucb_controller, ucb_US_controller, ucb_US_PP_controller, guided_learning_controller, thompson_controller, thompson_US_controller, hybrid_controller, round_robin_US_controller
from extra_controllers import impact_sampling_controller, greedy_controller
import pickle
import json
import sys
import cPickle
import inspect
from schema.job import Job
from schema.experiment import Experiment
import requests
from util import getLatestCheckpoint, split_examples, parse_answers, retrain, get_cost_of_action
from crowdjs_util import get_answers, upload_questions
from test_api import test_on_held_out_set, compute_performance_on_test_set
import uuid


@app.celery.task(name='restart')
def restart(job_id):
    job = Job.objects.get(id = job_id)
    
    checkpoint = getLatestCheckpoint(job_id)
    (task_information, budget) = pickle.loads(job.task_information)

    gather(task_information, budget, job_id, checkpoint)

#
#  Takes a set of task ids and gets all the answers from them 
#  Unlike parse_answers, assumes that the tasks are all completed.
#
#  only_sentence : return only the sentence, not all the other crowdsourced
#                  details
#

def gather_status(job_id, positive_types):
    job = Job.objects.get(id = job_id)
    
    checkpoint = getLatestCheckpoint(job_id)
    (task_information, budget) = pickle.loads(job.task_information)

    (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

    positive_examples, negative_examples = split_examples(
        task_ids, task_categories, positive_types, False)
    
    num_examples = len(positive_examples) + len(negative_examples)

    return [num_examples, positive_examples, negative_examples]



@app.celery.task(name='gathersim')
def gather_sim(task_information, budget, job_id, mturk_connection):

    training_examples = []
    training_positive_examples = []
    training_negative_examples = []
    training_labels = []
    task_ids = []
    task_categories = []
    costSoFar = 0

            
    job = Job.objects.get(id = job_id)

    
    #connection_string = str(requests.get(job.mturk_connection).content)
    #mturk_connection = cPickle.loads(connection_string)
    experiment = Experiment.objects.get(id = job.experiment_id)

    extra_job_state = {}

    while costSoFar < budget:
        print "Cost so far: %f" % costSoFar
        print "Number of training example and label batches: %d, %d" % (
            len(training_examples), len(training_labels))


        print "Deciding which category to do next"
        sys.stdout.flush()
        #Decide which category of task to do.
        (category_id, task_object, 
         num_hits, cost_of_next_task)  = get_next_batch(
            task_ids, task_categories, training_examples, training_labels,
            task_information, costSoFar, 
             extra_job_state, budget, job_id)


        task_id = str(uuid.uuid1())
        task_ids.append(task_id)
        task_categories.append(category_id)


        
        #number of workers per question is set in mturk layout
        (new_training_examples, 
         new_training_labels) = mturk_connection.create_hits(
            category_id, task_id,
            num_hits, task_object)
        

        #Bound the ratio of positives to negatives if so desired:
        if 'constant-ratio' in  job.control_strategy:
            (new_training_examples,
             new_training_labels) = bound_ratio_of_examples(
                 new_training_examples, new_training_labels)
        
        #update the cost
        print "Updating the Cost so far"
        print "adding" 
        print cost_of_next_task
        #print new_training_examples
        #print new_training_labels
        #print "THOSE WERE THE TRAINING EXMAPLES"
        sys.stdout.flush()

        costSoFar += cost_of_next_task


        num_new_positives = 0
        num_new_negatives = 0

        for (new_training_example, 
             new_training_label) in zip(new_training_examples,
                                        new_training_labels):
            if new_training_label == 1:
                num_new_positives += 1
                training_positive_examples.append(new_training_example)
            elif new_training_label == 0:
                num_new_negatives += 1
                training_negative_examples.append(new_training_example)
            else:
                print "This condition should not have happened"
                sys.stdout.flush()
                raise Exception


        experiment.statistics[job_id].append(
            (costSoFar, category_id, num_new_positives, num_new_negatives))

        training_examples.append(new_training_examples)
        training_labels.append(new_training_labels)


        cost_of_last_action = get_cost_of_action(
            task_categories[-1])
        costSoFar_before_last_action = (costSoFar - 
                                        cost_of_last_action)
        


        #The relevant threshold is 20.5 - (20.5  % 10) = 20
        #or 22 - (22/10) = 20
        relevant_threshold = costSoFar - (
            costSoFar % app.config[
                'EXPERIMENT_MEASUREMENT_INTERVAL'])
        

        #If and only if the cost BEFORE the last action is below the
        #relevant threshold, take a measurement. 
        if (('experiment_id' in job) and 
            (costSoFar_before_last_action < relevant_threshold)):
            #print "Computing current performance"
            #print training_positive_examples
            #print training_negative_examples
            #sys.stdout.flush() 

            precision, recall, f1 = compute_performance_on_test_set(
                job_id, task_ids, 
                experiment,
                training_positive_examples, training_negative_examples)
            experiment.learning_curves[job_id].append(
                (task_id, precision, recall, f1,
                 category_id, costSoFar))
            experiment.save()
                    

    task_id = task_ids[-1]
    category_id = task_categories[-1]
    
    
    precision, recall, f1 = compute_performance_on_test_set(
        job_id, task_ids, 
        experiment,
        training_positive_examples, training_negative_examples)
    experiment.learning_curves[job_id].append(
        (task_id, precision, recall, f1,
         category_id, costSoFar))
    experiment.save()
                    

    #Clean up and mark job as finished
    job.status = 'Finished'
    job.vocabulary = ""
    job.model_file = ""
    job.model_meta_file = ""
    job.save()


        
            
def get_next_batch(task_ids, task_categories,
                   training_examples, training_labels,
                   task_information, costSoFar, 
                   extra_job_state,
                   budget, job_id):
                      
    job = Job.objects.get(id = job_id)
    control_strategy = job.control_strategy
    
    print "Using the controller:"
    print control_strategy
    sys.stdout.flush()


    if control_strategy == 'guided-learning':
        return guided_learning_controller(task_ids,
                                      task_categories, training_examples,
                                      training_labels, task_information,
                                      costSoFar, budget, job_id)
    
    if control_strategy == 'round-robin-random-negatives':
        return round_robin_controller(task_ids,
                                      task_categories, training_examples,
                                      training_labels, task_information,
                                      costSoFar, budget, job_id)


    if control_strategy == 'round-robin-us':
        return round_robin_US_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    if control_strategy == 'seed3':
        return seed_controller(task_ids,
                               task_categories, training_examples,
                               training_labels, task_information,
                               costSoFar, budget, job_id)
    if control_strategy == 'seed3_us':
        return seed_US_controller(task_ids,
                               task_categories, training_examples,
                               training_labels, task_information,
                               costSoFar, budget, job_id)


    if control_strategy == 'round-robin-no-negate':
        return round_robin_no_negate_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    if control_strategy == 'uncertainty':
        return uncertainty_sampling_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)
    
    if control_strategy == 'impact':
        return impact_sampling_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)

    if control_strategy == 'label-only':
        return label_only_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)
    

    if control_strategy == 'label-only-us':
        return label_only_US_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)


    if control_strategy == 'greedy':
        return greedy_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar, budget, job_id)
    
    
    
    if control_strategy == 'ucb-us':
        return ucb_US_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar,
            extra_job_state,
            budget, job_id)

    if control_strategy == 'ucb-us-pp':
        return ucb_US_PP_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar,
            extra_job_state,
            budget, job_id)    

        
    if control_strategy == 'thompson-us':
        return thompson_US_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar,
            extra_job_state,
            budget, job_id)


    if control_strategy == 'hybrid-5e-1':
        return hybrid_controller(
            task_ids,
            task_categories, training_examples,
            training_labels, task_information,
            costSoFar,
            extra_job_state,
            budget, job_id,
            0.00005)
    




def bound_ratio_of_examples(examples, labels):

    negative_examples = []
    positive_examples = []
    
    for (example, label) in zip(examples, labels):
        if label == 1:
            positive_examples.append(example)
        else:
            negative_examples.append(example)
            
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = len(negative_examples)
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']
        

        
    selected_examples = []
    selected_labels = []

    for positive_example in positive_examples:
        selected_examples.append(positive_example)
        selected_labels.append(1)


        temp_num_negatives_wanted = num_negatives_wanted
        while temp_num_negatives_wanted > 0:
            if len(negative_examples) > 0:
                selected_examples.append(negative_examples.pop())
                selected_labels.append(0)
                temp_num_negatives_wanted -= 1
            else:
                break


    if len(positive_examples) == 0:
        if num_negatives_wanted > len(negative_examples):
            selected_examples += negative_examples
            selected_labels += [0 for i in range(len(negative_examples))]
        else:
            selected_examples += sample(negative_examples,num_negatives_wanted)
            expected_labels += [0 for i in range(num_negatives_wanted)]
            
    return selected_examples, selected_labels
            
