thesis-skew
==============
THIS IS RESEARCH CODE. USE AT YOUR OWN RISK.

## Development configuration
- Download and install Heroku
- Create and activate a Python virtual environment like conda.
- Dependencies are in requirements.txt

- Set up MongoDB, Redis, and RabbitMQ. A good way to do this is to create a free Heroku instance and then add the following free add-ons: mlab, heroku redis)
- Set up a RabbitMQ instance. (e.g., rabbitmq bigwig)

- Create a `.env` file in the root directory with the following lines:
```
MONGOLAB_URL = YOUR_CONNECTION_INFO_HERE
REDIS_URL = YOUR_CONNECTION_INFO_HERE
RABBITMQ_BIGWIG_URL = YOUR_CONNECTION_INFO_HERE
APP_SETTINGS='config.DevelopmentConfig'


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


# NO NEED TO DEFINE THE FOLLOWING TWO AWS KEYS UNLESS YOU WANT TO DO REAL ONLINE EXPERIMENTS, WHICH MAY OR MAY NOT WORK
AWS_ACCESS_KEY_ID = YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY = YOUR_ACS_SECRET_ACCESS_KEY_HERE


```


## Run instructions
- Run the application using `heroku local -p 7000` to start the web app on port 7000


## Usage
- Go to `index.html` to run an experiment.
- Go to 'analyze to view graphs


## Training Data
- `data/*_real` contains real generated data from crowdworkers for Health, Entertainment, Business, and Science headlines.
- `data/*_corpus` and `data/*_labeled_corpus` corpus contain data from the News Aggregator Data Set from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/News+Aggregator). 

## Test Sets
- Inside `data/test_data` are test sets. `test_strict_new_feature` is from Liu et al. (NAACL 2016). `testEvents` is from TAC-KBP dry run 2016. `self_generated` contains data sets created using this system.



