from app import db

class Gold_Extractor(db.DynamicDocument):

    name = db.StringField()

    model_file = db.FileField()
    model_meta_file = db.FileField()
    
    vocabulary = db.StringField()
    
    
    
    
