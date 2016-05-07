import os, sys, traceback
from flask import Flask
from flask.ext.cors import CORS
from flask.ext.restful import Api
from flask.ext.mail import Mail
import uuid
#import redis
#from rq import Worker, Queue, Connection
from rq import Queue

app = Flask(__name__)

app.config.from_object(os.environ['APP_SETTINGS'])
#app.redis = redis.StrictRedis.from_url(app.config['REDIS_URL'])


api = Api(app)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={"/assign_next_question": {"origins": "*"},
                            "/answers":  {"origins": "*"}})


print "Loading mail extension"
sys.stdout.flush()
mail = Mail(app)

print "Loading redis queue"
from worker import conn
app.rq = Queue(connection = conn)
    
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
    return "Welcome to Extremest Extraction v0.01!"
    

from api.taboo_api import *
api.add_resource(ComputeTabooApi, '/taboo') 
