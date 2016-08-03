extremest-extraction
==============

## Development configuration
- Create and activate a Python virtual environment. 
- Load application dependencies into the Python virtual environment by running `pip install -r requirements.txt`.
- Set up a MongoDB database. One option is to create a free Heroku instance with a MongoLab sandbox add-on.
- Create a `.env` file in the root directory with the following lines (substitute `db_user`, `db_password`, `host`, and `port` with details of your development MongoDB connection):
```
MONGOLAB_URI=mongodb://db_user:db_password@host:port
APP_SETTINGS='config.DevelopmentConfig'
```
- Create a production version `.production-env` that uses the production configuration.
```
APP_SETTINGS='config.Production'
```
- Set up a Redis instance. For Heroku, follow the instructions on https://devcenter.heroku.com/articles/heroku-redis
In your `.env` file, add:
```
REDIS_URL=redis://user:password@host:port
```

## Additional configuration
To set up Heroku environment, add the following buildpacks, using the toolbelt command `heroku buildpacks:add` or equivalent:

1. https://github.com/heroku/heroku-buildpack-python

## Run instructions
- Run the application using either `heroku local` (if using Heroku) or `./run.sh .env -b host:port`. Use the second option if you would like to see exceptions. 


## Testing instructions (No test framework yet)
- Use `heroku local -f Procfile.test` (if using Heroku) or
- Be sure to run both `./run_tests.sh .env` AND `./run_tests.sh .production-env` to test both dev and production environments.

## Usage
- Go to `/index.html` to define an event.


## Test Sets
- Inside `/data/test_data` are test sets. `test_strict_new_feature` is from Liu et al. (NAACL 2016). `testEvents` is from TAC-KBP dry run 2016. `self_generated` contains data sets created using this system. 