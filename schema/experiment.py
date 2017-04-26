from app import db

class Experiment(db.DynamicDocument):

    status = db.StringField()

    task_information = db.StringField()

    job_ids = db.ListField()
    control_strategy = db.StringField()

    control_strategy_configuration = db.StringField()

    num_runs = db.IntField()

    learning_curves = db.DictField()

    statistics = db.DictField()
    
    #Pickled dictionary where key is category id and value is
    #list of file names

    gpu_device_string = db.StringField()

    exceptions = db.ListField()

    dataset_skew = db.IntField()
    
    
