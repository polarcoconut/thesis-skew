from app import db

class UI_Test(db.DynamicDocument):

    task_id = db.StringField()
    task_category_id = db.StringField()
    current_hit_ids = db.ListField()


