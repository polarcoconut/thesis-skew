import os, sys, traceback
from flask import Flask
from flask.ext.cors import CORS
from flask.ext.restful import Api
from flask.ext.mail import Mail
from flask import render_template
import uuid
from rq import Queue
import redis
from boto.mturk.connection import MTurkConnection
from celery import Celery


app = Flask(__name__)

app.config.from_object(os.environ['APP_SETTINGS'])

#app.AWS_ACCESS_KEY_ID = app.config['AWS_ACCESS_KEY_ID']
#app.AWS_SECRET_ACCESS_KEY = app.config['AWS_SECRET_ACCESS_KEY']

#app.CROWDJS_API_KEY = app.config['CROWDJS_API_KEY']
#app.CROWDJS_REQUESTER_ID = app.config['CROWDJS_REQUESTER_ID']

#app.CROWDJS_GET_ANSWERS_URL = app.config['CROWDJS_GET_ANSWERS_URL']
#app.CROWDJS_SUBMIT_ANSWER_URL = app.config['CROWDJS_SUBMIT_ANSWER_URL']
#app.CROWDJS_PUT_TASK_URL =  app.config['CROWDJS_PUT_TASK_URL']
#app.CROWDJS_GET_TASK_DATA_URL =  app.config['CROWDJS_GET_TASK_DATA_URL']
#app.CROWDJS_PUT_TASK_DATA_URL =  app.config['CROWDJS_PUT_TASK_DATA_URL']
#app.CROWDJS_PUT_QUESTIONS_URL =  app.config['CROWDJS_PUT_QUESTIONS_URL']
#app.CROWDJS_RETURN_HIT_URL =  app.config['CROWDJS_RETURN_HIT_URL']
#app.CROWDJS_ASSIGN_URL =  app.config['CROWDJS_ASSIGN_URL']

#app.MTURK_HOST = app.config['MTURK_HOST']
#app.CONTROLLER = app.config['CONTROLLER']
#app.CONTROLLER_BATCH_SIZE = app.config['CONTROLLER_BATCH_SIZE']
#app.config['CONTROLLER_BATCH_SIZE'] = int(app.config['CONTROLLER_BATCH_SIZE'])
#app.CONTROLLER_APQ = app.config['CONTROLLER_APQ']
#app.config['CONTROLLER_APQ'] = int(app.config['CONTROLLER_APQ'])

#app.EXAMPLE_CATEGORIES = app.config['EXAMPLE_CATEGORIES']

api = Api(app)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={"/taboo": {"origins": "*"},
                            "/train":  {"origins": "*"}})


print "Loading mail extension"
sys.stdout.flush()
mail = Mail(app)

#print "Loading redis and redis queue"
#from worker import conn
#app.rq = Queue(connection = conn)
#app.redis = redis.StrictRedis.from_url(app.config['REDIS_URL'])

print "Loading Celery"
def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['REDIS_URL'],
                    broker=app.config['REDIS_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery
celery = make_celery(app)
app.celery = celery


print "Loading logging"
sys.stdout.flush()
if not app.debug:   
    import logging
    app.logger.addHandler(logging.StreamHandler())
    app.logger.setLevel(logging.ERROR)

print "Finished loading server."
sys.stdout.flush()



@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/status/<event_name>/<event_definition>/<event_pos_example_1>/<event_pos_example_1_trigger>/<event_pos_example_2>/<event_pos_example_2_trigger>/<event_pos_example_nearmiss>/<event_neg_example>/<event_neg_example_nearmiss>/<job_id>')
def status(event_name, event_definition, event_pos_example_1,
           event_pos_example_1_trigger, event_pos_example_2,
           event_pos_example_2_trigger, event_pos_example_nearmiss,
           event_neg_example, event_neg_example_nearmiss, job_id):
    return render_template(
        'status.html',
        event_name = event_name,
        event_definition = event_definition,
        event_pos_example_1 = event_pos_example_1,
        event_pos_example_1_trigger = event_pos_example_1_trigger,
        event_pos_example_2 = event_pos_example_2,
        event_pos_example_2_trigger = event_pos_example_2_trigger,
        event_pos_example_nearmiss = event_pos_example_nearmiss,
        event_neg_example = event_neg_example,
        event_neg_example_nearmiss = event_neg_example_nearmiss,
        job_id = job_id)


from api.taboo_api import ComputeTabooApi
api.add_resource(ComputeTabooApi, '/taboo') 

from api.train_api import TrainExtractorApi
api.add_resource(TrainExtractorApi, '/train')

from api.train_api import RestartExtractorApi
api.add_resource(RestartExtractorApi, '/restart')
