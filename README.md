thesis-skew
==============
THIS IS RESEARCH CODE. USE AT YOUR OWN RISK.

## Development configuration
- Download and install Heroku
- Create and activate a Python virtual environment like conda.
- Dependencies are in requirements.txt

- Set up MongoDB, Redis, and RabbitMQ. An easy way to do this is to create a free Heroku instance and then add the following free add-ons: mlab, heroku redis, rabbitmq bigwig)

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
- Go to `/index.html` to run an experiment. To reproduce graphs in the thesis, choose one of Health, Entertainment, Business, Science, Health_real, Entertainment_real, Business_real, Science_real, as the domain. Set the total budget to be 100, choose a control strategy, set the number of simulations to be 10, and set the skews to be 1,99,249,499,799,999. It will take awhile to run. If you want to run something fast, pick a strategy like round-robin-US, set the number ofsimulations to 1 and pick a single skew, like 99. Then click Experiment!

-Control strategies that correspond to algorithms in the thesis: MB-CB(Active) is ucb-us-constant-ratio. MB-T(Active) is thompson-us-constant-ratio. Label-Only(Active) is label-only-us-constant-ratio. Round-Robin is round-robin-us-constant-ratio. GL is Guided Learning. GL-Hybrid is hybrid-5e-1.

- Go to `/analyze` to view graphs. Skew-Analyze! will show average AUCs for each skew. Draw Individual Learning Curve will show the learning curves for each strategy for a single skew. Drawing graphs is slow, because the code is stupid. Be patient.


## Training Data
- `data/*_real` contains real generated data from crowdworkers for Health, Entertainment, Business, and Science headlines.
- `data/*_corpus` and `data/*_labeled_corpus` corpus contain data from the News Aggregator Data Set from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/News+Aggregator). 

## Test Sets
- Inside `data/test_data` are test sets. `test_strict_new_feature` is from Liu et al. (NAACL 2016). `testEvents` is from TAC-KBP dry run 2016. `self_generated` contains data sets created using this system.



