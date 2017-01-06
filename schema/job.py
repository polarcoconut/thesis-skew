from app import db

class Job(db.DynamicDocument):
    task_information = db.StringField()

    model_file = db.StringField()
    model_meta_file = db.StringField()

    vocabulary = db.StringField()
    num_training_examples_in_model = db.IntField()
    
    checkpoints = db.DictField()

    current_hit_ids = db.ListField()

    status = db.StringField()

    control_strategy = db.StringField()

    experiment_id = db.StringField()

    mturk_connection = db.StringField()

    exceptions = db.ListField()
    

    control_data = db.StringField()

    logging_data = db.StringField()

    unlabeled_corpus = db.StringField()

    gpu_device_string = db.StringField()
