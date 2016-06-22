import os

class Config(object):
    DEBUG = False
    TESTING = False

    #mongolab_uri = os.environ['MONGOLAB_URI'].split('/')
    #(dbuser, dbpass_host, port) = mongolab_uri[2].split(':')
    #(dbpass, host) = dbpass_host.split('@')
    #dbname = mongolab_uri[3]

    REDIS_URL = os.environ['REDIS_URL']

    #MONGODB_SETTINGS = {    
    #    'db': dbname,
    #    'host': host,
    #    'port': int(port),
    #    'username' : dbuser,
    #    'password' : dbpass}

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
    TABOO_THRESHOLD = int(os.environ['TABOO_THRESHOLD'])


    
    PRECISION_EXAMPLE_TASK = {
        'hit_layout_id' : os.environ['PRECISION_LAYOUT_ID'],
        'hit_type_id' :  os.environ['PRECISION_HITTYPE_ID'],
        'needs_data': True,
        'task_name' : 'Event Negation',
        'task_description' : 'Negate a sentence'}
    RECALL_EXAMPLE_TASK = {
        'hit_layout_id' : os.environ['RECALL_LAYOUT_ID'],
        'hit_type_id' :  os.environ['RECALL_HITTYPE_ID'],
        'needs_data' : False,
        'task_name' : 'Event Generation',
        'task_description' : 'Generate sentences'}

    
    EXAMPLE_CATEGORIES = [RECALL_EXAMPLE_TASK, PRECISION_EXAMPLE_TASK]

    CELERY_TIMEZONE = 'UTC'
    CELERY_IMPORTS = ['api.util', 'api.train']


    
class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    TESTING = True
    MTURK_HOST = 'mechanicalturk.sandbox.amazonaws.com'

class Production(Config):
    DEBUG = False
    DEVELOPMENT = False
    TESTING = False
    MTURK_HOST = 'mechanicalturk.amazonaws.com'
