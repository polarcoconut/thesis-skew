import os, sys, traceback
from flask import Flask
from flask.ext.cors import CORS
from flask.ext.restful import Api
from flask.ext.mail import Mail
import uuid
from rq import Queue
import redis
from boto.mturk.connection import MTurkConnection

app = Flask(__name__)

app.config.from_object(os.environ['APP_SETTINGS'])
app.AWS_ACCESS_KEY_ID = app.config['AWS_ACCESS_KEY_ID']
app.AWS_SECRET_ACCESS_KEY = app.config['AWS_SECRET_ACCESS_KEY']

app.CROWDJS_API_KEY = app.config['CROWDJS_API_KEY']
app.CROWDJS_GET_ANSWERS_URL = app.config['CROWDJS_GET_ANSWERS_URL']
app.CROWDJS_REQUESTER_ID = app.config['CROWDJS_REQUESTER_ID']

app.MTURK_HOST = app.config['MTURK_HOST']
app.CONTROLLER = app.config['CONTROLLER']
app.CONTROLLER_BATCH_SIZE = app.config['CONTROLLER_BATCH_SIZE']
app.CONTROLLER_APQ = app.config['CONTROLLER_APQ']

app.EXAMPLE_CATEGORIES = app.config['EXAMPLE_CATEGORIES']

api = Api(app)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={"/assign_next_question": {"origins": "*"},
                            "/answers":  {"origins": "*"}})


print "Loading mail extension"
sys.stdout.flush()
mail = Mail(app)

print "Loading redis and redis queue"
from worker import conn
app.rq = Queue(connection = conn)
app.redis = redis.StrictRedis.from_url(app.config['REDIS_URL'])

print "Setting up Mturk connection"
app.mturk = MTurkConnection(app.AWS_ACCESS_KEY_ID,
                            app.AWS_SECRET_ACCESS_KEY,
                            host=app.MTURK_HOST)

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
    html = "<p>Welcome to Extremest Extraction v0.01!</p>"
    html += '<form action="" method="post">Event: <input type="text"/><br/>'
    html += 'Event Definition: <input type="text"/></br/>'
    html += 'Event Good Example 1: <input type="text"/></br/>'
    html += 'Event Good Example 1 Trigger: <input type="text"/></br/>'
    html += 'Event Good Example 2: <input type="text"/></br/>'
    html += 'Event Good Example 2 Trigger: <input type="text"/></br/>'
    html += 'Event Bad Example 1: <input type="text"/></br/>'
    html += 'Event Negatve Good Example 1: <input type="text"/></br/>'
    html += 'Event Negatve Bad Example 1: <input type="text"/></br/>'
    html += '<input type="submit" value="Submit"></form>'
    return html




from api.taboo_api import *
api.add_resource(ComputeTabooApi, '/taboo') 

from api.train_api import *
api.add_resource(TrainExtractorApi, '/train')
