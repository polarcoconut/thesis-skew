from app import app
from boto.s3.connection import S3Connection
import uuid

def insert_model_into_s3(model_file_name, model_meta_file_name):
    
    s3 = S3Connection(app.config['AWS_ACCESS_KEY_ID'],
                                app.config['AWS_SECRET_ACCESS_KEY'])

    
    bucket = s3.get_bucket("extremest-extraction-temp-models")

    model_key_name = str(uuid.uuid4())
    model_meta_key_name = str(uuid.uuid4())

    model_key = bucket.new_key(model_key_name)

    model_key.set_contents_from_filename(model_file_name)

    model_meta_key = bucket.new_key(model_meta_key_name)

    model_meta_key.set_contents_from_filename(model_meta_file_name)
        

    model_url = model_key.generate_url(360)
    model_meta_url = model_meta_key.generate_url(360)

    return model_url, model_meta_url

