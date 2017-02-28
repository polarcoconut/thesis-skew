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

#print "Flush the cache"
app.redis = redis.Redis.from_url(app.config['REDIS_URL'])
#app.redis.flushdb()

db = MongoEngine(app)



print "Loading Celery"
def make_celery(app):
    #celery = Celery(app.import_name, backend=app.config['REDIS_URL'],
    #                broker=app.config['REDIS_URL'])
    celery = Celery(app.import_name, broker=app.config['RABBITMQ_BIGWIG_URL'])
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
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)

print "Finished loading server."
sys.stdout.flush()



@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/analyze')
def analyze_endpoint():
    return render_template('analyze.html')


@app.route('/status/<job_id>')
def status(job_id):
    return render_template(
        'status.html',
        job_id = job_id)

@app.route('/experiment_status/<experiment_id>')
def experiment_status(experiment_id):
    return render_template(
        'experiment_status.html',
        experiment_id = experiment_id)

@app.route('/test/<task_id>/<task_category_id>')
def test(task_id, task_category_id):
    return render_template(
        'test.html',
        task_id = task_id,
        task_category_id = task_category_id)


from api.taboo_api import ComputeTabooApi
api.add_resource(ComputeTabooApi, '/taboo') 

from api.train_api import GatherExtractorApi
api.add_resource(GatherExtractorApi, '/gather')

from api.train_api import RestartApi
api.add_resource(RestartApi, '/restart')

from api.train_api import PauseApi
api.add_resource(PauseApi, '/pause')

from api.train_api import JobStatusApi
api.add_resource(JobStatusApi, '/status')

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

from api.util_api import GetJobInfoApi
api.add_resource(GetJobInfoApi, '/get_job_info')

from api.util_api import TestGenerateUIApi
api.add_resource(TestGenerateUIApi, '/test_generate_ui')

from api.util_api import TestModifyUIApi
api.add_resource(TestModifyUIApi, '/test_modify_ui')

from api.util_api import TestLabelUIApi
api.add_resource(TestLabelUIApi, '/test_label_ui')

from api.util_api import ChangeBudgetApi
api.add_resource(ChangeBudgetApi, '/change_budget')

from api.util_api import CleanUpApi
api.add_resource(CleanUpApi, '/cleanup')


from api.experiment_api import ExperimentApi
api.add_resource(ExperimentApi, '/experiment')

from api.experiment_api import ExperimentStatusApi
api.add_resource(ExperimentStatusApi, '/get_experiment_status')

from api.experiment_api import ExperimentAnalyzeApi
api.add_resource(ExperimentAnalyzeApi, '/analyze_experiment')

from api.experiment_api import AllExperimentAnalyzeApi
api.add_resource(AllExperimentAnalyzeApi, '/analyze_all_experiments')

