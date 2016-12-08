from app import app
import requests
import pickle
import sys, traceback
import time


def submit_answer(task_id, worker_id, question_name, answer):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    submit_answer_request = {}
    submit_answer_request['requester_id'] = app.config['CROWDJS_REQUESTER_ID']
    submit_answer_request['task_id'] = task_id;
    submit_answer_request['question_name'] = question_name;
    submit_answer_request['worker_id'] = worker_id;
    submit_answer_request['worker_source'] = 'mturk';
    submit_answer_request['value'] = answer;

    while True:
        try:
            r = requests.put(app.config['CROWDJS_SUBMIT_ANSWER_URL'],
                             headers=headers,
                             json=submit_answer_request)
            print "Here is the response"
            print r.text
            sys.stdout.flush()
            return True
        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue
        



def get_next_assignment(task_id, worker_id):
    assign_crowdjs_url = app.config['CROWDJS_ASSIGN_URL']
    assign_crowdjs_url += '?worker_id=';
    assign_crowdjs_url += worker_id; 
    assign_crowdjs_url += '&worker_source=mturk';
    assign_crowdjs_url += '&task_id=';
    assign_crowdjs_url += task_id;
    assign_crowdjs_url += '&requester_id=';
    assign_crowdjs_url += app.config['CROWDJS_REQUESTER_ID']
    assign_crowdjs_url += '&preview=false';
    assign_crowdjs_url += '&strategy=';
    assign_crowdjs_url+= 'min_answers';

    while True:
        try:
            r = requests.get(assign_crowdjs_url)
            data = r.json()
            
            return data

        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue


    
def get_task_data(task_id):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    task_crowdjs_url = app.config['CROWDJS_GET_TASK_DATA_URL']
    task_crowdjs_url += '?task_id=%s' % task_id
    task_crowdjs_url += '&requester_id=%s' % app.config[
        'CROWDJS_REQUESTER_ID']

    while True:
        try:
            r = requests.get(task_crowdjs_url, headers=headers)
            data = r.json()['data']
            
            return data

        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue

    
def get_answers(task_id):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    answers_crowdjs_url = app.config['CROWDJS_GET_ANSWERS_URL']
    answers_crowdjs_url += '?task_id=%s' % task_id
    answers_crowdjs_url += '&requester_id=%s' % app.config[
        'CROWDJS_REQUESTER_ID']
    answers_crowdjs_url += '&completed=True'

    while True:
        try:
            r = requests.get(answers_crowdjs_url, headers=headers)
            answers = r.json()
            
            return answers

        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue


def get_questions(task_id):
    #headers = {'Authentication-Token': app.config['CROWDJS_API_KEY']}
    questions_crowdjs_url = app.config['CROWDJS_GET_QUESTIONS_URL'] % task_id

    while True:
        try:
            r = requests.get(questions_crowdjs_url)
            questions = r.json()
            return questions

        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue



def get_answers_for_question(question_ids):
    answers_crowdjs_url = app.config['CROWDJS_GET_ANSWERS_FOR_QUESTION_URL']
    answers_crowdjs_url += '?completed=True'

    for question_id in question_ids:
        answers_crowdjs_url += '&question_ids=%s' % question_id

    while True:
        try:
            r = requests.get(answers_crowdjs_url)
            answers = r.json()
            return answers
    
        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue


    

def upload_questions(task):
    headers = {'Authentication-Token': app.config['CROWDJS_API_KEY'],
               'content_type' : 'application/json'}


    while True:
        try:
            r = requests.put(app.config['CROWDJS_PUT_TASK_URL'],
                             headers=headers,
                             json=task)
            print "Here is the response"
            print r.text
            sys.stdout.flush()
            
            response_content = r.json()
            task_id = response_content['task_id']    
            
            return task_id

        except Exception:
            print "Exception while communicating with crowdjs:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue


def make_labeling_crowdjs_task(examples, expected_labels,
                               task_information):

    category = app.config['EXAMPLE_CATEGORIES'][2]

    question_data_format = ''
    for information in task_information:
        question_data_format += "%s\t"
    question_data_format += "%d\t"
    question_data_format += "%s"
        
    questions = []
    
    for i, example, expected_label in zip(range(len(examples)), examples,
                                          expected_labels):
        new_question_information = task_information + (expected_label, example)
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
        'answers_per_question' : app.config['CONTROLLER_GENERATE_BATCH_SIZE'],
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
    for i, training_example in zip(range(len(examples)), examples):
        new_question_information = task_information + (training_example,)
        new_question_data =  question_data % new_question_information
        question = {
            'question_name': 'Precision Question %d' % i,
            'question_description': 'Precision Question %d' % i,
            'question_data': new_question_data,
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'answers_per_question' : app.config[
                'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']}
        questions.append(question)
        

    task = {'task_name': category['task_name'],
            'task_description': category['task_description'],
            'requester_id' : app.config['CROWDJS_REQUESTER_ID'],
            'data' : pickle.dumps(
                {'not': 1}),
            'assignment_duration' : app.config['ASSIGNMENT_DURATION'],
            'questions' : questions}

    return task
