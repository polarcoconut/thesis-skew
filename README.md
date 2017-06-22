thesis-skew
==============
THIS IS RESEARCH CODE. USE AT YOUR OWN RISK.

## Development configuration
- Create and activate a Python virtual environment like virtualenv or conda.
- Download and set up Heroku
- Dependencies are in requirements.txt

- Set up MongoDB, Redis, and RabbitMQ. A good way to do this is to create a free Heroku instance and then add the following add-ons:
- Set up a MongoDB database (e.g., mlab) . 
- Set up a Redis instance. (e.g., heroku redis)
- Set up a RabbitMQ instance. (e.g., rabbitmq bigwig)

- Create a `.env` file in the root directory with the following lines:
```
MONGOLAB_URL = YOUR_CONNECTION_INFO_HERE
REDIS_URL = YOUR_CONNECTION_INFO_HERE
RABBITMQ_BIGWIG_URL = YOUR_CONNECTION_INFO_HERE
APP_SETTINGS='config.DevelopmentConfig'
AWS_ACCESS_KEY_ID = YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY = YOUR_ACS_SECRET_ACCESS_KEY_HERE

BATCH_SIZE = 50

CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE = 1

CONTROLLER_LABELS_PER_QUESTION = 1


TABOO_THRESHOLD = 0
ASSIGNMENT_DURATION = 120

AL_THRESHOLD = 100

LABEL_TASK_PRICE = 0.03
GENERATE_POS_TASK_PRICE = 0.15
#GENERATE_NEG_TASK_PRICE = 0.10
GENERATE_NEG_TASK_PRICE = 0.00

MODEL = LR
#MODELS CAN BE ONE OF CNN, CRF, LR

```


## Run instructions
- Run the application using either `heroku local` (if using Heroku) or `./run. 


## Usage
- Go to `/index.html` to run an experiment.
- Go to '/analyze to view graphs

## Test Sets
- Inside `/data/test_data` are test sets. `test_strict_new_feature` is from Liu et al. (NAACL 2016). `testEvents` is from TAC-KBP dry run 2016. `self_generated` contains data sets created using this system. 