import os, sys, traceback
from flask import Flask
from flask.ext.cors import CORS
from flask.ext.restful import Api
from flask.ext.mail import Mail
from flask.ext.mongoengine import MongoEngine
from flask import render_template
import uuid
from boto.mturk.connection import MTurkConnection
from celery import Celery
import redis

app = Flask(__name__)

app.config.from_object(os.environ['APP_SETTINGS'])

api = Api(app)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={"/taboo": {"origins": "*"},
                            "/train":  {"origins": "*"}})


print "Loading mail extension"
sys.stdout.flush()
mail = Mail(app)

print "Loading redis and mongo"
#from worker import conn
#app.rq = Queue(connection = conn)
#app.redis = redis.Redis.from_url(app.config['REDIS_URL'])
db = MongoEngine(app)

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

from api.train_api import GatherExtractorApi
api.add_resource(GatherExtractorApi, '/gather')

from api.train_api import RestartApi
api.add_resource(RestartApi, '/restart')

from api.train_api import PauseApi
api.add_resource(PauseApi, '/pause')

from api.train_api import GatherStatusApi
api.add_resource(GatherStatusApi, '/gather_status')

from api.train_api import RetrainExtractorApi
api.add_resource(RetrainExtractorApi, '/retrain')

from api.train_api import RetrainStatusApi
api.add_resource(RetrainStatusApi, '/retrain_status')


from api.test_api import TestExtractorApi
api.add_resource(TestExtractorApi, '/test')

from api.test_api import CrossValidationExtractorApi
api.add_resource(CrossValidationExtractorApi, '/cv')

from api.util_api import MoveJobsFromRedisToMongoApi
api.add_resource(MoveJobsFromRedisToMongoApi, '/move')
