from app import db

class Job(db.DynamicDocument):
    task_information = db.StringField()

    #model = db.StringField()
    #model_meta = db.StringField()
    model = db.StringField()
    model_meta = db.StringField()
    model_file = db.BinaryField()
    model_meta_file = db.BinaryField()

    vocabulary = db.StringField()
    num_training_examples_in_model = db.IntField()
    
    checkpoints = db.DictField()

