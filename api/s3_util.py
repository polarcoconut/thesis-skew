from app import app
from boto.s3.connection import S3Connection
import uuid
import boto

def insert_model_into_s3(model_file_name, model_meta_file_name):
    
    s3 = S3Connection(app.config['AWS_ACCESS_KEY_ID'],
                                app.config['AWS_SECRET_ACCESS_KEY'])

    
    bucket = s3.get_bucket("extremest-extraction-temp-models")

    model_key_name = str(uuid.uuid1())
    model_meta_key_name = str(uuid.uuid1())

    model_key = bucket.new_key(model_key_name)

    model_key.set_contents_from_filename(model_file_name)

    model_meta_key = bucket.new_key(model_meta_key_name)

    model_meta_key.set_contents_from_filename(model_meta_file_name)
        

    model_key.make_public()
    model_meta_key.make_public()

    model_url = model_key.generate_url(3600000)
    model_meta_url = model_meta_key.generate_url(3600000)

    return model_url, model_meta_url


def insert_connection_into_s3(connection_pickle):
    
    s3 = S3Connection(app.config['AWS_ACCESS_KEY_ID'],
                                app.config['AWS_SECRET_ACCESS_KEY'])

    
    bucket = s3.get_bucket("extremest-extraction-temp")
    bucket_location = bucket.get_location()
    if bucket_location:
        conn = boto.s3.connect_to_region(bucket_location)
        bucket = conn.get_bucket("extremest-extraction-temp")

    connection_key_name = str(uuid.uuid1())

    connection_key = bucket.new_key(connection_key_name)

    connection_key.set_contents_from_string(connection_pickle)

    connection_key.make_public()
    connection_url = connection_key.generate_url(3600000)

    return connection_url

