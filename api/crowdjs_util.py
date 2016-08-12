from app import app
import requests
import pickle
import sys

def get_answers(task_id):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    answers_crowdjs_url = app.config['CROWDJS_GET_ANSWERS_URL']
    answers_crowdjs_url += '?task_id=%s' % task_id
    answers_crowdjs_url += '&requester_id=%s' % app.config[
        'CROWDJS_REQUESTER_ID']
    answers_crowdjs_url += '&completed=True'

    r = requests.get(answers_crowdjs_url, headers=headers)

    answers = r.json()

    return answers

def get_questions(task_id):
    #headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    questions_crowdjs_url = app.config['CROWDJS_GET_QUESTIONS_URL'] % task_id
    r = requests.get(questions_crowdjs_url)
    questions = r.json()
    return questions

def get_answers_for_question(question_id):
    answers_crowdjs_url = app.config['CROWDJS_GET_ANSWERS_FOR_QUESTION_URL'] % question_id
    answers_crowdjs_url += '?completed=True'
    
    r = requests.get(answers_crowdjs_url)
    answers = r.json()
    return answers
    

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

def make_labeling_crowdjs_task(examples, task_information):

    category = app.config['EXAMPLE_CATEGORIES'][2]

    question_data_format = ''
    for information in task_information:
        question_data_format += "%s\t"
    question_data_format += "%s"
        
    questions = []
    
    for i, example in zip(range(len(examples)), examples):
        new_question_information = task_information + (example,)
        new_question_data =  question_data_format % new_question_information
        question = {
            'question_name': 'Labeling Question %d' % i,
            'question_description': 'Labeling Question %d' % i,
            'question_data': new_question_data,
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'answers_per_question' : app.config[
                'CONTROLLER_LABELS_PER_QUESTION'],
            'unique_workers' : True}
        questions.append(question)


    task = {'task_name': category['task_name'],
            'task_description': category['task_description'],
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'data' : 'no_data',
            'assignment_duration' : app.config['ASSIGNMENT_DURATION'],
            'questions' : questions}
    
    return task


def make_recall_crowdjs_task(task_information):
    (event_name, event_definition,
     event_good_example_1,
     event_good_example_1_trigger,
     event_good_example_2,
     event_good_example_2_trigger,
     event_bad_example_1,
     event_negative_good_example_1,
     event_negative_bad_example_1) = task_information

    category = app.config['EXAMPLE_CATEGORIES'][0]
    question_data = ''
    for information in task_information:
        question_data += "%s\t"
        
    question_data =  question_data % task_information
    questions = []
    
    #Just have 1 question
    question = {
        'question_name': 'Recall Question',
        'question_description': 'Recall Question',
        'question_data': question_data,
        'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
        'answers_per_question' : (app.config['CONTROLLER_BATCH_SIZE'] *
                                  app.config['CONTROLLER_APQ']),
        'unique_workers' : False}
    questions.append(question)
    
    task = {'task_name': category['task_name'],
            'task_description': category['task_description'],
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'data' : pickle.dumps({event_good_example_1_trigger:1,
                                   event_good_example_2_trigger:1}),
            'assignment_duration' : app.config['ASSIGNMENT_DURATION'],
            'questions' : questions}

    return task

def make_precision_crowdjs_task(examples, task_information):
    category = app.config['EXAMPLE_CATEGORIES'][1]

    question_data = ''
    for information in task_information:
        question_data += "%s\t"
    question_data += "%s"


    questions = []
    last_batch = examples
    for i, training_example in zip(range(len(last_batch)), last_batch):
        new_question_information = task_information + (training_example,)
        new_question_data =  question_data % new_question_information
        question = {
            'question_name': 'Precision Question %d' % i,
            'question_description': 'Precision Question %d' % i,
            'question_data': new_question_data,
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'answers_per_question' : app.config['CONTROLLER_APQ']}
        questions.append(question)
        

    task = {'task_name': category['task_name'],
            'task_description': category['task_description'],
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'data' : pickle.dumps(
                {'not': 1}),
            'assignment_duration' : app.config['ASSIGNMENT_DURATION'],
            'questions' : questions}

    return task
