import time
from app import app
from controllers import greedy_controller
import pickle
import json
import sys

from schema.job import Job
from mturk_util import delete_hits, create_hits
from util import getLatestCheckpoint, split_examples, parse_answers
from crowdjs_util import get_answers, upload_questions


@app.celery.task(name='restart')
def restart(job_id):
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
    gather(task_information, budget, job_id, checkpoint)

#
#  Takes a set of task ids and gets all the answers from them 
#  Unlike parse_answers, assumes that the tasks are all completed.
#
#  only_sentence : return only the sentence, not all the other crowdsourced
#                  details
#

def gather_status(job_id, positive_types):
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
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
            #print "loading task_id %s" % task_id
            #sys.stdout.flush()
            
            answers = parse_answers(task_id, task_category_id)
            new_examples, new_labels = answers
            training_examples.append(new_examples)
            training_labels.append(new_labels)

            
    job = Job.objects.get(id = job_id)

    if costSoFar >= budget:
        task_id = task_ids[-1]
        category_id = task_categories[-1]
        
        #Check if the task is complete
        answers = parse_answers(task_id, category_id, wait_until_batch_finished = len(job.current_hit_ids))

        if answers:
            print "Delete any existing leftover hits from turk"
            if 'current_hit_ids' in job and len(job.current_hit_ids) > 0:
                delete_hits(job.current_hit_ids)
            job.status = 'Finished'
            job.save()
        return True
            
    else:
        print "Cost so far: %d" % costSoFar
        print "Number of training example batches and label batches: %d, %d" % (
            len(training_examples), len(training_labels))


        #If we have task_ids, wait for the last one to complete.
        if len(task_ids) > 0:
            task_id = task_ids[-1]
            category_id = task_categories[-1]

            #Check if the task is complete
            answers = parse_answers(task_id, category_id, wait_until_batch_finished = len(job.current_hit_ids))

            if answers:
                new_examples, new_labels = answers
                sys.stdout.flush()
                
                training_examples.append(new_examples)
                training_labels.append(new_labels)
            else:
                print "Task not complete yet"
                sys.stdout.flush()
                return False

        print "Delete any existing leftover hits from turk"
        if 'current_hit_ids' in job and len(job.current_hit_ids) > 0:
            delete_hits(job.current_hit_ids)
        
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
        hit_ids = create_hits(category_id, task_id,
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
                      

    print "Using the controller:"
    print app.config['CONTROLLER']
    if app.config['CONTROLLER'] == 'greedy':
        return round_robin_controller(task_ids,
                                      task_categories, training_examples,
                                      training_labels, task_information,
                                      costSoFar, budget, job_id)
