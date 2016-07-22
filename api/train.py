import time
from app import app
from controllers import greedy_controller
import requests
import pickle
import json
import sys
from ml.train_cnn import trainCNN
from ml.extractors.cnn_core.test import test_cnn
from schema.job import Job
from mturk_util import delete_hits, create_hits

def upload_questions(task):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY'],
               'content_type' : 'application/json'}

    r = requests.put(app.config['CROWDJS_PUT_TASK_URL'],
                     headers=headers,
                     json=task)
    print "Here is the response"
    print app.config['CROWDJS_API_KEY']
    print r.text
    sys.stdout.flush()
    
    response_content = r.json()
    task_id = response_content['task_id']    
        
    return task_id



def getLatestCheckpoint(job_id):
    #read the latest checkpoint
    job = Job.objects.get(id = job_id)


    timestamps = job.checkpoints.keys()

        
    most_recent_timestamp = max([int(x) for x in timestamps])

    checkpoint = job.checkpoints[str(most_recent_timestamp)]
    (task_information, budget) = pickle.loads(job.task_information)

    return (task_information, budget, checkpoint)


@app.celery.task(name='restart')
def restart(job_id):
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
    gather(task_information, budget, job_id, checkpoint)

def split_examples(task_ids, task_categories, positive_types = []):
    positive_examples = []
    negative_examples = []
    for task_id, task_category in zip(task_ids,task_categories):
        answers = parse_answers(task_id, task_category, False, positive_types)
        print answers
        new_examples, new_labels = answers
        for new_example, new_label in zip(new_examples, new_labels):
            if new_label == 1:
                positive_examples.append(new_example)
            else:
                negative_examples.append(new_example)
    print positive_examples
    print negative_examples
    return positive_examples, negative_examples

def gather_status(job_id, positive_types):
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
    (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

    positive_examples, negative_examples = split_examples(
        task_ids, task_categories, positive_types)
    
    num_examples = len(positive_examples) + len(negative_examples)

    return [num_examples, positive_examples, negative_examples]

@app.celery.task(name='retrain')
def retrain(job_id, positive_types):
    print "Training a CNN"
    sys.stdout.flush()
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
    (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)


    training_positive_examples, training_negative_examples = split_examples(
        task_ids[0:-2],
        task_categories[0:-2],
        positive_types)
    
    model_file_name, vocabulary = trainCNN(
        training_positive_examples, training_negative_examples)

    

    model_file_handle = open(model_file_name, 'rb')
    model_binary = model_file_handle.read()

    model_meta_file_handle = open("{}.meta".format(model_file_name), 'rb')
    model_meta_binary = model_meta_file_handle.read()

    print "Saving the model"
    print job_id
    sys.stdout.flush()

    job = Job.objects.get(id=job_id)
    job.vocabulary = pickle.dumps(vocabulary)
    job.model_file = model_binary
    job.model_meta_file = model_meta_binary
    print "Model saved"
    sys.stdout.flush()

    job.num_training_examples_in_model = (
        len(training_positive_examples) + len(training_negative_examples))

    print "training saved"
    sys.stdout.flush()

    job.save()

    print "Job modified"
    sys.stdout.flush()

    model_file_handle.close()
    model_meta_file_handle.close()

    print "file handles closed"
    sys.stdout.flush()
    
    return True


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
        for task_id, task_category in zip(task_ids[0:-1],
                                          task_categories[0:-1]):
            print "loading task_id %s" % task_id
            sys.stdout.flush()
            
            answers = parse_answers(task_id, task_category)
            new_examples, new_labels = answers
            training_examples.append(new_examples)
            training_labels.append(new_labels)

            
    
    while costSoFar < budget:
        print "Cost so far: %d" % costSoFar
        print "Number of training example batches and label batches: %d, %d" % (
            len(training_examples), len(training_labels))
        print training_examples

        #make a checkpoint
        checkpoint = pickle.dumps((task_ids, task_categories, costSoFar))
        job = Job.objects.get(id = job_id)
        job.checkpoints[str(int(time.time()))] = checkpoint
        job.save()

        #If we have task_ids, wait for the last one to complete.
        if len(task_ids) > 0:
            task_id = task_ids[-1]
            category = task_categories[-1]
            #Wait for the task to complete.
            while True:
                print "Sleeping"
                sys.stdout.flush()

                time.sleep(10)
                #Check if the task is complete
                answers = parse_answers(task_id, category)

                if answers:
                    new_examples, new_labels = answers
                    print "New examples"
                    print new_examples
                    sys.stdout.flush()
                    
                    training_examples.append(new_examples)
                    training_labels.append(new_labels)
                    break
                print "Task not complete yet"
                sys.stdout.flush()

        print "Delete any existing leftover hits from turk"
        if 'current_hit_ids' in job and len(job.current_hit_ids) > 0:
            delete_hits(job.current_hit_ids)
        
        print "Deciding which category to do next"
        print "Task categories so far:"
        print task_categories
        sys.stdout.flush()
        #Decide which category of task to do.
        category, task_object, num_hits  = get_next_batch(
            task_categories, training_examples, training_labels,
            task_information)

        
        print "hit type to do next: %s" % category['task_name']
        print "Uploading task to CrowdJS"
        sys.stdout.flush()

        #Upload task to CrowdJS
        task_id = upload_questions(task_object)
        task_ids.append(task_id)
        task_categories.append(category)

        print "Task Ids:"
        print task_ids
        print "Uploading task to Mturk"
        sys.stdout.flush()

        #Upload assignments onto MTurk
        #number of workers per question is set in mturk layout
        hit_ids = create_hits(category, task_id,
                              num_hits)
        job.current_hit_ids = hit_ids
        job.save()
        
        print "Hit IDs:"
        print hit_ids
        sys.stdout.flush()

        #update the cost
        print "Updating the Cost so far"
        costSoFar += (app.config['CONTROLLER_BATCH_SIZE'] *
                      app.config['CONTROLLER_APQ'])
        




def get_answers(task_id, category):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    answers_crowdjs_url = app.config['CROWDJS_GET_ANSWERS_URL']
    answers_crowdjs_url += '?task_id=%s' % task_id
    answers_crowdjs_url += '&requester_id=%s' % app.config[
        'CROWDJS_REQUESTER_ID']
    r = requests.get(answers_crowdjs_url, headers=headers)

    answers = r.json()

    return answers

def parse_answers(task_id, category, wait_until_batch_finished=True,
                  positive_types = []):

    answers = get_answers(task_id, category)

    print "Number of answers"
    print len(answers)
    sys.stdout.flush()
            
    if wait_until_batch_finished and (len(answers) <
        app.config['CONTROLLER_BATCH_SIZE'] * app.config['CONTROLLER_APQ']):
        return None

    examples = []
    labels = []
    
    #Data structure of checkpoints were updated so old checkpoints may not have
    #an id in them.
    if (('id' in category and category['id'] == 0) or
        category['task_name'] == 'Event Generation'):
        for answer in answers:
            value = answer['value']
                
            value = value.split('\t')
            sentence = value[0]
            trigger = value[1]

            print sentence
            print value[4]
            print 'general' in positive_types
            
            if len(value) > 2:
                if value[2] == 'Yes':                
                    past = True
                else:
                    past = False
                    
                if value[3] == 'Yes':
                    future = True
                else:
                    future = False
                    
                if value[4] == 'Yes':
                    general = True
                else:
                    general = False
            
            

                if 'past' in positive_types and past:
                        labels.append(1)                
                elif 'future' in positive_types and future:
                        labels.append(1)
                elif 'general' in positive_types and general:
                        labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(1)

            examples.append(sentence)

    elif (('id' in category and category['id'] == 1) or
          category['task_name'] == 'Event Negation'):
        for answer in answers:
            value = answer['value']
            value = value.split('\t')
            sentence = value[0]
            previous_sentence_not_example_of_event = value[1]

            if len(value) > 2:
                if value[2] == 'failing':                
                    failing = True
                else:
                    failing = False
            
                if 'general' in positive_types:
                    if failing:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            else:
                labels.append(0)
            examples.append(sentence)

    print (examples, labels)
    return examples, labels
        
            
def get_next_batch(task_categories, training_examples, training_labels,
                   task_information):
                      

    print "Using the controller:"
    print app.config['CONTROLLER']
    if app.config['CONTROLLER'] == 'greedy':
        return greedy_controller(task_categories, training_examples,
                                 training_labels,
                                 task_information)
