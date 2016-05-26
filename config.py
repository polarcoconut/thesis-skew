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

    AWS_ACCESS_KEY_ID = 'AKIAIOGNWDMLQIYOKRQQ'
    AWS_SECRET_ACCESS_KEY = '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'


    #CROWDJS STUFF
    CROWDJS_API_KEY = 'seomthing'
    CROWDJS_REQUESTER_ID = 'asdfasdfasdf'
    CROWDJS_GET_ANSWERS_URL = 'http://crowdjs.heroku.com/answers?'
    CROWDJS_PUT_TASK_URL =  'http://crowdjs.heroku.com/tasks'
    CROWDJS_PUT_QUESTIONS_URL =  'http://crowdjs.heroku.com/questions'



    
    PRECISION_EXAMPLE_TASK = {'hit_layout_id' : 'alskdfadf',
                              'needs_data': True,
                              'task_name' : 'Event Negation',
                              'task_description' : 'Negate a sentence'}
    RECALL_EXAMPLE_TASK = {'hit_layout_id' : 'aklsjdlka',
                           'needs_data' : False,
                           'task_name' : 'Event Generation',
                           'task_description' : 'Generate sentences'}

    
    EXAMPLE_CATEGORIES = [RECALL_EXAMPLE_TASK, PRECISION_EXAMPLE_TASK]


    #greedy, fixed
    CONTROLLER = 'greedy'
    CONTROLLER_BATCH_SIZE = 100
    CONTROLLER_APQ = 2

    
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
