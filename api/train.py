import time

from boto.mturk.question import ExternalQuestion
from boto.mturk.price import Price
from boto.mturk.qualification import Qualifications
from boto.mturk.qualification import PercentAssignmentsApprovedRequirement
from boto.mturk.qualification import NumberHitsApprovedRequirement
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters
from app import app
from controllers import greedy_controller
import requests
import pickle

def upload_questions(task):
    headers = {'Authentication-Token': app.CROWDJS_API_KEY,
               'content_type' : 'application/json'}

    r = requests.put(app.CROWDJS_PUT_TASK_URL, headers=headers,
                     json=task)

    response_content = r.json()
    task_id = response_content['task_id']    
        
    return task_id

def create_hits(hit_layout_id, task_id, num_hits):


    layout_params = LayoutParameters()
    for hit_num in range(num_hits):
        layout_params.add(
            LayoutParameter('questionNumber', '%s' % hit_num))
        layout_params.add(
            LayoutParameter('task_id', '%s' % task_id))
        layout_params.add(
            LayoutParameter('requester_id', '%s'% app.CROWDJS_REQUESTER_ID))

        hits = mturk.create_hit(hit_layout = hit_layout_id,
                                layout_params = layout_params)

    return [hit.HITId for hit in hits]

def train(task_information, budget):
    
    #task_ids is a list of the task_ids that have been assigned
    #training exampes is a list of lists of examples from each task
    task_ids = []
    training_examples = []
    training_labels = []
    task_categories = []
    
    costSoFar = 0
    while costSoFar < budget:
        
        #Decide which category of task to do.
        category, task_object  = get_next_batch(
            task_categories, training_examples, training_labels,
            task_information)

        hit_layout_id = category['hit_layout_id']
        
        #Upload task to CrowdJS
        task_id = upload_questions(hit_layout_id, task_object)
        task_ids.append(task_id)
        task_categories.append(category)
        
        #Upload assignments onto MTurk
        create_hits(hit_layout_id, task_id,
                    app.CONTROLLER_BATCH_SIZE * app.CONTROLLER_APQ)
        

        #Wait for the task to complete.
        while True:
            print "Sleeping"
            time.sleep(10)
            #Check if the task is complete
            answers = parse_answers(task_id, category)
            if answers:
                new_examples, new_labels = answers
                training_examples.append(new_examples)
                training_labels.append(new_labels)
                break

        #Now that the task is complete, make a checkpoint
        checkpoint = pickle.dumps((task_ids, task_categories))
        app.redis.set(str(int(time.time())), checkpoint)


def parse_answers(task_id, category):
    headers = {'Authentication-Token': app.CROWDJS_API_KEY}
    answers_crowdjs_url += 'task_id=%s' % task_id
    answers_crowdjs_url += '&requester_id=%s' % requester_id
    r = requests.get(app.CROWDJS_GET_ANSWERS_URL, headers=headers)        
    answers = r.json()

    if len(answers) <= app.CONTROLLER_BATCH_SIZE * app.CONTROLLER_APQ:
        return None
    examples = []
    labels = []
    if category['name'] == 'Event Generation':
        for answer in answers:
            value = answer['value']
            value = value.split('\t')
            sentence = value[0]
            trigger = value[1]
            examples.append(sentence)
            labels.append(1)
    elif category['name'] == 'Event Negation':
        for answer in answers:
            value = answer['value']
            examples.append(value)
            labels.append(0)

    return examples, labels
        
            
def get_next_batch(task_categories, training_examples, training_labels,
                   task_information):
                      
                                                        
    if app.controller == 'greedy':
        greedy_controller(answers, training_examples, training_labels,
                          task_information)
