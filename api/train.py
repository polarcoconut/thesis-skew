import time
from app import app
from controllers import round_robin_controller, uncertainty_sampling_controller, impact_sampling_controller, label_only_controller
import pickle
import json
import sys
import cPickle
import inspect
from schema.job import Job
from schema.experiment import Experiment
import requests
from util import getLatestCheckpoint, split_examples, parse_answers, retrain
from crowdjs_util import get_answers, upload_questions
from test_api import test_on_held_out_set, compute_performance_on_test_set

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



@app.celery.task(name='gather')
def gather(task_information, budget, job_id, checkpoint = None):

    training_examples = []
    training_labels = []
    task_ids = []
    task_categories = []
    costSoFar = 0

    #task_ids is a list of the task_ids that have been assigned by crowdjs
    #training exampes is a list of lists of examples from each task
    if checkpoint:
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        print "loading checkpoint..."
        for task_id, task_category_id in zip(task_ids[0:-1],
                                          task_categories[0:-1]):
            
            answers = parse_answers(task_id, task_category_id)
            new_examples, new_labels = answers
            training_examples.append(new_examples)
            training_labels.append(new_labels)

            
    job = Job.objects.get(id = job_id)

    if 'experiment_id' in job:
        
        #print "CONNECTION_STRING"
        #print requests.get(job.mturk_connection).text
        #sys.stdout.flush() 
        
        connection_string = str(requests.get(job.mturk_connection).content)
        mturk_connection = cPickle.loads(connection_string)
        experiment = Experiment.objects.get(id = job.experiment_id)
    else:
        mturk_connection = MTurk_Connection_Real()

    if costSoFar >= budget:
        if not job.status == 'Finished':
            task_id = task_ids[-1]
            category_id = task_categories[-1]
            
            #Check if the task is complete
            answers = parse_answers(
                task_id, category_id, 
                wait_until_batch_finished = len(job.current_hit_ids))

            if answers:
                print "Delete any existing leftover hits from turk"
                if 'current_hit_ids' in job and len(job.current_hit_ids) > 0:
                    mturk_connection.delete_hits(job.current_hit_ids)
                job.status = 'Finished'
                job.save()
                if 'experiment_id' in job:
                    print "Computing current performance"
                    sys.stdout.flush() 
                    precision, recall, f1 = compute_performance_on_test_set(
                        job_id, task_ids, 
                        experiment)
                    experiment.learning_curves[job_id].append(
                        (task_id, precision, recall, f1,
                         category_id, costSoFar))
                    experiment.save()
                    


        else:
            print "job became finished after getting queued, so do nothing."
        return True
            
    else:
        print "Cost so far: %d" % costSoFar
        print "Number of training example and label batches: %d, %d" % (
            len(training_examples), len(training_labels))


        #If we have task_ids, wait for the last one to complete.
        if len(task_ids) > 0:
            task_id = task_ids[-1]
            category_id = task_categories[-1]

            #Check if the task is complete
            answers = parse_answers(
                task_id, category_id, 
                wait_until_batch_finished = len(job.current_hit_ids))

            if answers:
                new_examples, new_labels = answers
                sys.stdout.flush()
                
                training_examples.append(new_examples)
                training_labels.append(new_labels)

                if 'experiment_id' in job:
                    print "Computing current performance"
                    sys.stdout.flush() 
                    precision, recall, f1 = compute_performance_on_test_set(
                        job_id, task_ids, 
                        experiment)
                    experiment.learning_curves[job_id].append(
                        (task_id, precision, recall, f1,
                         category_id, costSoFar))
                    experiment.save()

            else:
                print "Task not complete yet"
                sys.stdout.flush() 
                return False

        print "Delete any existing leftover hits from turk"
        if 'current_hit_ids' in job and len(job.current_hit_ids) > 0:
            mturk_connection.delete_hits(job.current_hit_ids)
        
        print "Deciding which category to do next"
        sys.stdout.flush()
        #Decide which category of task to do.
        category_id, task_object, num_hits, cost_of_next_task  = get_next_batch(
            task_ids, task_categories, training_examples, training_labels,
            task_information, costSoFar, budget, job_id)

        
        print "hit type to do next: %s" % category_id
        print "Uploading task to CrowdJS"
        sys.stdout.flush()

        #Upload task to CrowdJS
        task_id = upload_questions(task_object)
        task_ids.append(task_id)
        task_categories.append(category_id)

        print "Task Ids:"
        print task_ids
        print "Uploading task to Mturk"
        sys.stdout.flush()

        #Upload assignments onto MTurk
        #number of workers per question is set in mturk layout
        hit_ids = mturk_connection.create_hits(category_id, task_id,
                              num_hits)
        job.current_hit_ids = hit_ids
        job.save()
        
        print "Hit IDs:"
        print hit_ids
        sys.stdout.flush()

        #update the cost
        print "Updating the Cost so far"
        costSoFar += cost_of_next_task
        
        #make a checkpoint
        checkpoint = pickle.dumps((task_ids, task_categories, costSoFar))
        job.checkpoints[str(int(time.time()))] = checkpoint
        job.save()





        
            
def get_next_batch(task_ids, task_categories,
                   training_examples, training_labels,
                   task_information, costSoFar, budget, job_id):
                      
    job = Job.objects.get(id = job_id)
    control_strategy = job.control_strategy
    
    print "Using the controller:"
    print control_strategy
    sys.stdout.flush()

    if control_strategy == 'round-robin':
        return round_robin_controller(task_ids,
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
    
