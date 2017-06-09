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


    MODEL = os.environ['MODEL']

    REDIS_URL = os.environ['REDIS_URL']

    RABBITMQ_BIGWIG_URL = os.environ['RABBITMQ_BIGWIG_URL']
    
    MONGODB_SETTINGS = {    
        'db': dbname,
        'host': host,
        'port': int(port),
        'username' : dbuser,
        'password' : dbpass,
        'connectTimeoutMS': 0,
        'socketTimeoutMS' : 0}

    SECRET_KEY = 'super-secret'
    SECURITY_REGISTERABLE = True
    SECURITY_PASSWORD_HASH = 'sha512_crypt'
    SECURITY_PASSWORD_SALT = 'abcde'

    MAIL_SUPPRESS_SEND = True


    #IT used to be 2.
    UCB_EXPLORATION_CONSTANT = 1.0
    #It used to be 1.0
    UCB_SMOOTHING_PARAMETER = 1.0

    CROWDJS_API_KEY = os.environ['CROWDJS_API_KEY']
    CROWDJS_REQUESTER_ID = os.environ['CROWDJS_REQUESTER_ID']

    CROWDJS_BASE_URL = os.environ['CROWDJS_BASE_URL']
    CROWDJS_GET_ANSWERS_URL = CROWDJS_BASE_URL + '/answers'
    CROWDJS_SUBMIT_ANSWER_URL = CROWDJS_BASE_URL + '/answers'
    CROWDJS_SUBMIT_BATCH_ANSWER_URL = CROWDJS_BASE_URL + '/answers2'
    CROWDJS_GET_QUESTIONS_URL = CROWDJS_BASE_URL + '/tasks/%s/questions'
    CROWDJS_GET_ANSWERS_FOR_QUESTION_URL = CROWDJS_BASE_URL + '/questions/answers'

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

    BATCH_SIZE = int(os.environ[
        'BATCH_SIZE'])
    CONTROLLER_GENERATE_BATCH_SIZE = int(os.environ[
        'BATCH_SIZE'])
    CONTROLLER_LABELING_BATCH_SIZE = int(os.environ[
        'BATCH_SIZE'])

    CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE = int(os.environ[
        'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
    CONTROLLER_LABELS_PER_QUESTION = int(
        os.environ['CONTROLLER_LABELS_PER_QUESTION'])
    TABOO_THRESHOLD = int(os.environ['TABOO_THRESHOLD'])
    ASSIGNMENT_DURATION = int(os.environ['ASSIGNMENT_DURATION'])

    AL_THRESHOLD = int(os.environ['AL_THRESHOLD'])

    #Every task is an action.
    
    PRECISION_EXAMPLE_TASK = {
        'id' : 1,
        'needs_data': True,
        'price' : float(os.environ['GENERATE_NEG_TASK_PRICE']),
        'hit_html' : open('tasks/negateevent.html').read(),
        'task_name' : 'Event Modification',
        'task_description' : 'Modify a sentence so that it either expresses a different event than the one it currently expresses or it negates the event.'}
    RECALL_EXAMPLE_TASK = {
        'id' : 0,
        'needs_data' : False,
        'price' : float(os.environ['GENERATE_POS_TASK_PRICE']),
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

    STRATEGY_NAMES = {            
        'seed3' : 'Seed-PositiveLabeling-Bounded-Ratio',
        'seed3_us' : 'Seed-ActiveLabeling',
        'seed3_us_constant_ratio' : 'Seed-ActiveLabeling-Bounded-Ratio',
        'round-robin-constant-ratio' : 'Round-Robin-Bounded-Ratio',
        'round-robin-half-constant-ratio' : 'Round-Robin-Half-Bounded-Ratio',
        'round-robin-us-constant-ratio' : 'Round-Robin-US-Bounded-Ratio',
        'label-only-constant-ratio' : 'RandomLabel-Only-Bounded-Ratio',
        'label-only' : 'RandomLabel-Only',
        'ucb-constant-ratio' : 'UCB(GenPos-LabelPosBR)',
        'ucb-us' : 'UCB(GenPos-LabelActive)',
        'ucb-us-pp' : 'UCB(GenPos-LabelPosUSBR)',
        'ucb-us-constant-ratio' : 'UCB(GenPos-LabelActiveBR)',
        'thompson-constant-ratio' : 'Thompson(GenPos-LabelPosBR)',
        'thompson-us' : 'Thompson(GenPos-LabeActive)',
        'thompson-us-constant-ratio' : 'Thompson(GenPos-LabelActiveBR)',
        'guided-learning': 'Guided-Learning',
        'hybrid-5e-1' : 'Hybrid-5e-1'}

    AVAILABLE_UCB_CONSTANTS = [2.0]

    CELERY_TIMEZONE = 'UTC'
    CELERY_IMPORTS = ['api.util', 'api.train', 'periodic_tasks']
    CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml']
    BROKER_POOL_LIMIT = 0
    CELERYD_PREFETCH_MULTIPLIER = 1
    CELERY_ACKS_LATE = True

    UCI_NEWS_AGGREGATOR_HEALTH = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/health_corpus'
    UCI_NEWS_AGGREGATOR_HEALTH_LABELED = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/health_labeled_corpus'

    UCI_NEWS_AGGREGATOR_ENT = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/ent_corpus'
    UCI_NEWS_AGGREGATOR_ENT_LABELED = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/ent_labeled_corpus'

    UCI_NEWS_AGGREGATOR_BUS = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/bus_corpus'
    UCI_NEWS_AGGREGATOR_BUS_LABELED = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/bus_labeled_corpus'

    UCI_NEWS_AGGREGATOR_SCI = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/sci_corpus'
    UCI_NEWS_AGGREGATOR_SCI_LABELED = 'https://s3-us-west-2.amazonaws.com/extremest-extraction-uci-news-aggregator-data/sci_labeled_corpus'
    

    EXPERIMENT_WORKER_ACC = 1.0

    EXPERIMENT_MEASUREMENT_INTERVAL = 10
    
    #If num_negatives_per_positive is negative, then use the native
    #skew.
    NUM_NEGATIVES_PER_POSITIVE = 3

    
class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    TESTING = True
    MTURK_HOST = 'mechanicalturk.sandbox.amazonaws.com'
    MTURK_EXTERNAL_SUBMIT = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    #CELERY_REDIS_MAX_CONNECTIONS = 5
    QUALIFICATIONS = Qualifications([LocaleRequirement('EqualTo', 'US')])

    CELERYBEAT_SCHEDULE = {
        #'delete_temp_files': {
        #    'task': 'delete_temp_files',
        #    'schedule': timedelta(seconds=86400),
        #    'args': ()
        #},

        'run_gather': {
            'task': 'run_gather',
            'schedule': timedelta(seconds=20),
            'args': ()
        },
    }

    #TACKBP_NW_09_CORPUS_URL = 'https://s3-us-west-2.amazonaws.com/tac-kbp-2009/sentences.meta-sentencesonly-no-liu-et-al-naacl2016-test-set-2k'
    TACKBP_NW_09_CORPUS_URL = 'https://s3-us-west-2.amazonaws.com/tac-kbp-2009/sentences.meta-sentencesonly-no-liu-et-al-naacl2016-test-set-270k'


    

    
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
        #'delete_temp_files': {
        #    'task': 'delete_temp_files',
        #    'schedule': timedelta(seconds=86400),
        #    'args': ()
        #},
        'run_gather': {
            'task': 'run_gather',
            'schedule': timedelta(seconds=60),
            'args': ()
        },
    }
    TACKBP_NW_09_CORPUS_URL = 'https://s3-us-west-2.amazonaws.com/tac-kbp-2009/sentences.meta-sentencesonly-no-liu-et-al-naacl2016-test-set-270k'

