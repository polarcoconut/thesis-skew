from app import db

class Experiment(db.DynamicDocument):

    task_information = db.StringField()

    job_ids = db.ListField()
    control_strategy = db.StringField()
    num_runs = db.IntField()

    precisions = db.ListField()
    recalls = db.ListField()
    fscores = db.ListField()

    task_ids_for_simulation = db.ListField()
    

    test_set = db.IntField()
    
    
