from app import db

class Experiment(db.DynamicDocument):

    task_information = db.StringField()

    job_ids = db.ListField()
    control_strategy = db.StringField()

    control_strategy_configuration = db.StringField()

    num_runs = db.IntField()

    learning_curves = db.DictField()
    
    #Pickled dictionary where key is category id and value is
    #list of file names
    files_for_simulation = db.StringField()
    

    test_set = db.IntField()
    
    
