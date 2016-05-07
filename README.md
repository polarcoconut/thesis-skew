extremest-extraction
==============

## Development configuration
- Create and activate a Python virtual environment. https://github.com/yyuu/pyenv and https://github.com/yyuu/pyenv-virtualenv are useful tools for this.
- Load application dependencies into the Python virtual environment by running `pip install -r requirements.txt`.
- This repository contains git submodules. Run `git submodule init` and `git submodule update` to fetch these.
- Set up a MongoDB database. One option is to create a free Heroku instance with a MongoLab sandbox add-on.
- Create a `.env` file in the root directory with the following lines (substitute `db_user`, `db_password`, `host`, and `port` with details of your development MongoDB connection; install zmdp using the zmdp submodule, and substitute `zmdp` with the alias for running the ZMDP solver. ):
```
MONGOLAB_URI=mongodb://db_user:db_password@host:port
APP_SETTINGS='config.DevelopmentConfig'
ZMDP_ALIAS=zmdp
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
To set up Heroku environment to run ZMDP, add the following buildpacks, using the toolbelt command `heroku buildpacks:add` or equivalent:

1. https://github.com/heroku/heroku-buildpack-python
2. https://github.com/jbragg/heroku-buildpack-zmdp.git

## Run instructions
- Run the application using either `heroku local` (if using Heroku) or `./run.sh .env -b host:port`. Use the second option if you would like to see exceptions. 


## Testing instructions
- Use `heroku local -f Procfile.test` (if using Heroku) or
- Be sure to run both `./run_tests.sh .env` AND `./run_tests.sh .production-env` to test both dev and production environments.
- The `test/` folder also contains folders with more tests that include end-to-end workflow tests as well as more unit tests. To run these, read the README inside the desired `*_workflow/` folder.

## Usage
- First, create an account and login by going to `server_url/register` and `server_url/login`. You will receive an API Token as well as a requester_id.
- Next, make a PUT request to `server_url/tasks` to insert your task (consisting of 1 or more questions) into the database. This step requires your credentials.
- To query the next question a worker should answer, make a GET request to `server_url/assign_next_question`.
- To insert an answer into the database, make a PUT request to `server_url/answers`.
- See the documentation for more details about how to make the requests.


## Workflow Management
- When inserting a task, you may include a function (as a string) that will be called every time an answer is submitted by a worker. You can use this function to create new questions and remove old questions. Keep in mind this function will always be called, unless you specify otherwise (when making a put request to `/answers`), even if the answer submitted is for a question that has already been removed.
- Inside this function, the function may modify two variables: `new_questions`, which is a python list that contains a list of new questions you want added to your task, and `old_question_budget`, an integer that you can set that allows you to mutate the budget of the question that was just answered. (Setting it to 0 effectively prevents workers from seeing this question).
- Your function should NOT read or write from the database in any way, nor should it make any assumptions about the rest of the code. It should only modify the 2 variables specified above using code that does not depend on crowdjs.