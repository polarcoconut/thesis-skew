import os
from boto.mturk.qualification import Qualifications
from boto.mturk.qualification import PercentAssignmentsApprovedRequirement
from boto.mturk.qualification import NumberHitsApprovedRequirement
from boto.mturk.qualification import LocaleRequirement
from datetime import timedelta

class Config(object):
    DEBUG = False
    TESTING = False

    mongolab_uri = os.environ['MONGOLAB_URI'].split('/')
    (dbuser, dbpass_host, port) = mongolab_uri[2].split(':')
    (dbpass, host) = dbpass_host.split('@')
    dbname = mongolab_uri[3]

    REDIS_URL = os.environ['REDIS_URL']

    MONGODB_SETTINGS = {    
        'db': dbname,
        'host': host,
        'port': int(port),
        'username' : dbuser,
        'password' : dbpass}

    SECRET_KEY = 'super-secret'
    SECURITY_REGISTERABLE = True
    SECURITY_PASSWORD_HASH = 'sha512_crypt'
    SECURITY_PASSWORD_SALT = 'abcde'

    MAIL_SUPPRESS_SEND = True

    CROWDJS_API_KEY = os.environ['CROWDJS_API_KEY']
    CROWDJS_REQUESTER_ID = os.environ['CROWDJS_REQUESTER_ID']

    CROWDJS_BASE_URL = os.environ['CROWDJS_BASE_URL']
    CROWDJS_GET_ANSWERS_URL = CROWDJS_BASE_URL + '/answers'
    CROWDJS_SUBMIT_ANSWER_URL = CROWDJS_BASE_URL + '/answers'
    CROWDJS_GET_QUESTIONS_URL = CROWDJS_BASE_URL + '/tasks/%s/questions'
    CROWDJS_GET_ANSWERS_FOR_QUESTION_URL = CROWDJS_BASE_URL + '/questions/%s/answers'

    CROWDJS_PUT_TASK_URL =  CROWDJS_BASE_URL + '/tasks'
    CROWDJS_GET_TASK_DATA_URL =  CROWDJS_BASE_URL + '/task_data'
    CROWDJS_PUT_TASK_DATA_URL =  CROWDJS_BASE_URL + '/task_data'
    CROWDJS_PUT_QUESTIONS_URL =  CROWDJS_BASE_URL + '/questions'
    CROWDJS_RETURN_HIT_URL =  CROWDJS_BASE_URL + '/requeue'
    CROWDJS_ASSIGN_URL =  CROWDJS_BASE_URL + '/assign_next_question'

    EE_BASE_URL = os.environ['EE_BASE_URL']
    SUBMIT_TABOO_URL = EE_BASE_URL + '/taboo'

    
    AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
    
    CONTROLLER = os.environ['CONTROLLER']
    CONTROLLER_BATCH_SIZE = int(os.environ['CONTROLLER_BATCH_SIZE'])
    CONTROLLER_APQ = int(os.environ['CONTROLLER_APQ'])
    CONTROLLER_LABELS_PER_QUESTION = int(
        os.environ['CONTROLLER_LABELS_PER_QUESTION'])
    TABOO_THRESHOLD = int(os.environ['TABOO_THRESHOLD'])
    ASSIGNMENT_DURATION = int(os.environ['ASSIGNMENT_DURATION'])

    
    PRECISION_EXAMPLE_TASK = {
        'id' : 1,
        'needs_data': True,
        'price' : 0.10,
        'hit_html' : open('tasks/negateevent.html').read(),
        'task_name' : 'Event Modification',
        'task_description' : 'Modify a sentence so that it either expresses a different event than the one it currently expresses or it negates the event.'}
    RECALL_EXAMPLE_TASK = {
        'id' : 0,
        'needs_data' : False,
        'price' : 0.15,
        'hit_html' : open('tasks/generate.html').read(),
        'task_name' : 'Event Generation',
        'task_description' : 'Provide a sentence that is an example of a given event.'}

    LABEL_EXAMPLE_TASK = {
        'id' : 2,
        'needs_data' : False,
        'price' : float(os.environ['LABEL_TASK_PRICE']),
        'hit_html' : open('tasks/label.html').read(),
        'task_name' : 'Event Labeling',
        'task_description' : 'Determine whether a sentence is an example of a given event.'}

    
    EXAMPLE_CATEGORIES = [RECALL_EXAMPLE_TASK, PRECISION_EXAMPLE_TASK,
                          LABEL_EXAMPLE_TASK]

    CELERY_TIMEZONE = 'UTC'
    CELERY_IMPORTS = ['api.util', 'api.train', 'periodic_tasks']
    CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml']
    BROKER_POOL_LIMIT = 0


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    TESTING = True
    MTURK_HOST = 'mechanicalturk.sandbox.amazonaws.com'
    MTURK_EXTERNAL_SUBMIT = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    #CELERY_REDIS_MAX_CONNECTIONS = 5
    QUALIFICATIONS = Qualifications([LocaleRequirement('EqualTo', 'US')])

    CELERYBEAT_SCHEDULE = {
        'run_gather': {
            'task': 'run_gather',
            'schedule': timedelta(seconds=20),
            'args': ()
        },
    }
class Production(Config):
    DEBUG = False
    DEVELOPMENT = False
    TESTING = False
    MTURK_HOST = 'mechanicalturk.amazonaws.com'
    MTURK_EXTERNAL_SUBMIT = 'https://www.mturk.com/mturk/externalSubmit'
    QUALIFICATIONS = Qualifications(
        [NumberHitsApprovedRequirement('GreaterThan', 500),
         PercentAssignmentsApprovedRequirement('GreaterThan', 95),
         LocaleRequirement('EqualTo', 'US')])

    CELERYBEAT_SCHEDULE = {
        'run_gather': {
            'task': 'run_gather',
            'schedule': timedelta(seconds=60),
            'args': ()
        },
    }
