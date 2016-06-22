import time

from app import app
from boto.mturk.question import ExternalQuestion
from boto.mturk.price import Price
from boto.mturk.qualification import Qualifications
from boto.mturk.qualification import PercentAssignmentsApprovedRequirement
from boto.mturk.qualification import NumberHitsApprovedRequirement
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters
from boto.mturk.connection import MTurkConnection
from controllers import greedy_controller
import requests
import pickle
import json
import sys
import redis


def upload_questions(task, config):
    headers = {'Authentication-Token': config['CROWDJS_API_KEY'],
               'content_type' : 'application/json'}

    r = requests.put(config['CROWDJS_PUT_TASK_URL'],
                     headers=headers,
                     json=task)
    print "Here is the response"
    print config['CROWDJS_API_KEY']
    print r.text
    sys.stdout.flush()
    
    response_content = r.json()
    task_id = response_content['task_id']    
        
    return task_id

def create_hits(hit_type_id, hit_layout_id, task_id, num_hits, config):

    print "Connecting to Turk host at"
    print config['MTURK_HOST']
    sys.stdout.flush()
    
    mturk = MTurkConnection(config['AWS_ACCESS_KEY_ID'],
                            config['AWS_SECRET_ACCESS_KEY'],
                            host=config['MTURK_HOST'])

    print "Uploading %d hits to turk" % num_hits
    hits = []
    for hit_num in range(num_hits):
        layout_params = LayoutParameters()

        #layout_params.add(
        #    LayoutParameter('questionNumber', '%s' % hit_num))
        layout_params.add(
            LayoutParameter('task_id', '%s' % task_id))
        layout_params.add(
            LayoutParameter('requester_id', '%s'%
                            config['CROWDJS_REQUESTER_ID']))

        layout_params.add(
            LayoutParameter(
                'task_data_url', '%s'%
                config['CROWDJS_GET_TASK_DATA_URL']))
        layout_params.add(
            LayoutParameter(
                'submit_answer_url', '%s'%
                config['CROWDJS_SUBMIT_ANSWER_URL']))
        layout_params.add(
            LayoutParameter(
                'compute_taboo_url', '%s'%
                config['SUBMIT_TABOO_URL']))
        layout_params.add(
            LayoutParameter(
                'return_hit_url', '%s'%
                config['CROWDJS_RETURN_HIT_URL']))
        layout_params.add(
            LayoutParameter(
                'assign_url', '%s'%
                config['CROWDJS_ASSIGN_URL']))

        layout_params.add(
            LayoutParameter(
                'taboo_threshold', '%s'%
                config['TABOO_THRESHOLD']))


        print layout_params
        sys.stdout.flush()
        
        hit = mturk.create_hit(
            hit_type= hit_type_id,
            hit_layout = hit_layout_id,
            layout_params = layout_params)[0]
        
        hits.append(hit.HITId)
        
    return hits

@app.celery.task(name='restart')
def restart(task_information, config):
    #read the latest checkpoint
    checkpoint_redis = redis.StrictRedis.from_url(config['REDIS_URL'])
    key = pickle_dumps(task_information)
    timestamps = checkpoint_redis.hkeys(key)

    most_recent_timestamp = max([int(x) for x in timestamps])

    checkpoint = checkpoint_redis.hmget(key, most_recent_timestamp)

    train(task_information, budget, config, checkpoint)
          
def test(task_information):
    #read the latest checkpoint
    checkpoint = pickle.dumps((task_ids, task_categories))
    checkpoint_redis = redis.StrictRedis.from_url(config['REDIS_URL'])
    checkpoint_redis.set(str(int(time.time())), checkpoint)

@app.celery.task(name='train')
def train(task_information, budget, config, checkpoint = None):

    #task_ids is a list of the task_ids that have been assigned by crowdjs
    #training exampes is a list of lists of examples from each task
    if checkpoint:
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        print "loading checkpoint..."
        for task_id, task_category in zip(task_ids[0:-1],
                                          task_categories[0:-1]):
            print "loading task_id" % task_id
            sys.stdout.flush()
            
            answers = parse_answers(task_id, category, config)
            new_examples, new_labels = answers
            training_examples.append(new_examples)
            training_labels.append(new_labels)

        
    else:
        task_ids = []
        task_categories = []
        costSoFar = 0
        training_examples = []
        training_labels = []

    
    
    while costSoFar < budget:
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
                answers = parse_answers(task_id, category, config)
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

        print "cost so far: %d" % costSoFar
        print "Deciding which category to do next"
        sys.stdout.flush()
        #Decide which category of task to do.
        category, task_object  = get_next_batch(
            task_categories, training_examples, training_labels,
            task_information, config)

        hit_layout_id = category['hit_layout_id']
        hit_type_id = category['hit_type_id']
        
        print "hit_layout_id to do next: %s" % hit_layout_id
        print "Uploading task to CrowdJS"
        sys.stdout.flush()

        #Upload task to CrowdJS
        task_id = upload_questions(task_object, config)
        task_ids.append(task_id)
        task_categories.append(category)

        print "Task Ids:"
        print task_ids
        print "Uploading task to Mturk"
        sys.stdout.flush()

        #Upload assignments onto MTurk
        #number of workers per question is set in mturk layout
        hit_ids = create_hits(
            hit_type_id,
            hit_layout_id, task_id,
            int(config['CONTROLLER_BATCH_SIZE']),
            config)

        print "Hit IDs:"
        print hit_ids
        sys.stdout.flush()

        #update the cost
        costSoFar += (config['CONTROLLER_BATCH_SIZE'] * config['CONTROLLER_BATCH_SIZE'])
        
        #make a checkpoint
        checkpoint = pickle.dumps((task_ids, task_categories, costSoFar))
        checkpoint_redis = redis.StrictRedis.from_url(config['REDIS_URL'])
        #checkpoint_redis.set(str(int(time.time())), checkpoint)
        checkpoint_redis.hset(pickle.dumps(task_information),
                              str(int(time.time())),
                              checkpoint)





def parse_answers(task_id, category, config):
    headers = {'Authentication-Token': config['CROWDJS_API_KEY']}
    answers_crowdjs_url = config['CROWDJS_GET_ANSWERS_URL']
    answers_crowdjs_url += '?task_id=%s' % task_id
    answers_crowdjs_url += '&requester_id=%s' % config['CROWDJS_REQUESTER_ID']
    r = requests.get(answers_crowdjs_url, headers=headers)
    print "Response from getting answers"
    print r.text
    sys.stdout.flush()

    answers = r.json()

    if (len(answers) <
        config['CONTROLLER_BATCH_SIZE'] * config['CONTROLLER_APQ']):
        return None
    examples = []
    labels = []
    print "Answers"
    if category['task_name'] == 'Event Generation':
        for answer in answers:
            value = answer['value']
            print value
            sys.stdout.flush()
                
            value = value.split('\t')
            sentence = value[0]
            trigger = value[1]
            examples.append(sentence)
            labels.append(1)
    elif category['task_name'] == 'Event Negation':
        for answer in answers:
            value = answer['value']
            examples.append(value)
            labels.append(0)

    return examples, labels
        
            
def get_next_batch(task_categories, training_examples, training_labels,
                   task_information, config):
                      
                                                        
    if config['CONTROLLER'] == 'greedy':
        return greedy_controller(task_categories, training_examples,
                                 training_labels,
                                 task_information, config)
