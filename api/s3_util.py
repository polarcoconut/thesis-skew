from app import app
from boto.s3.connection import S3Connection
import uuid
import boto
from random import sample, shuffle
import os

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

    return model_url, model_meta_url, model_key_name, model_meta_key_name

def insert_crf_model_into_s3(model_file_name):
    
    s3 = S3Connection(app.config['AWS_ACCESS_KEY_ID'],
                                app.config['AWS_SECRET_ACCESS_KEY'])

    
    bucket = s3.get_bucket("extremest-extraction-temp-models")

    model_key_name = str(uuid.uuid1())

    model_key = bucket.new_key(model_key_name)

    model_key.set_contents_from_filename(model_file_name)
        

    model_key.make_public()

    model_url = model_key.generate_url(3600000)

    return model_url



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





def generate_dataset(interested_category,
                     num_of_negatives_per_positive):
    
    ###
    #THIS IS THE NEWS AGGREGATOR DATASET FROM THE UCI ML REPO
    ###
    input_file = open('temp_datasets/newsCorpora.csv', 'r')


    categories = {'bus' : 'b', 
                  'sci':'t', 
                  'ent' : 'e', 
                  'health' : 'm',
                  'bus_real' : 'b',
                  'sci_real' : 't',
                  'ent_real' : 'e',
                  'health_real' : 'm'}
    num_examples_per_category = {'b' : 0, 't' : 0,
                                 'e' : 0, 'm' : 0}
    interested_category = categories[interested_category]

    use_real_generated_data = False
    if (interested_category == 'bus_real' or
        interested_category == 'sci_real' or
        interested_category == 'ent_real' or
        interested_category == 'health_real'):
        use_real_generated_data = True

    data = {}

    num_sentences = 0

    positive_examples = []
    negative_examples = []

    for row in input_file:
        #print row
        row = row.split('\t')
        news_id = row[0]
        title = row[1]
        url = row[2]
        publisher = row[3]
        category = row[4]
        story_id = row[5]
        hostname = row[6]
        timestamp = row[7]

        if story_id not in data:
            data[story_id] = []

        data[story_id].append(title)
        num_sentences += 1

        if category == interested_category:
            positive_examples.append(title)
        else:
            negative_examples.append(title)

    print "Number of stories"
    print len(data.keys())
    print "Number of sentences"
    print num_sentences

    print "Number of positive examples"
    print len(positive_examples)
    print "Number of negative examples"
    print len(negative_examples)

    shuffle(positive_examples)
    shuffle(negative_examples)


    #Sample 2000 positives for "generated data"
    num_test_examples = 100
    num_crowd_positives = 2000


    #IN A PREVIOUS EXPERIMENT, we
    #sampled 3500 positives and threw the rest away.
    #num_positive_examples = 3500

    #First figure out how many positives we can include
    #We want to maximize num_positive_examples, such that
    # num_positive_examples < len(positive_examples) and
    # ratio * num_positive_examples < len(negative_examples)



    if use_real_generated_data:
        num_positive_examples = min(
            len(negative_examples) / num_of_negatives_per_positive,
            len(positive_examples))


        positive_examples = sample(positive_examples, num_positive_examples)


        real_generated_data = open('temp_datasets/%s' % interested_category,
                                   'r')

        crowd_positive_examples = []
        for row in real_generate_data:
            crowd_positive_examples.append(row)
        real_generated_data.close()
        
        corpus_positive_examples = positive_examples[
            0:len(positive_examples) - num_test_examples]
        testing_positive_examples = positive_examples[
            len(positive_examples) - num_test_examples:]
        
    else:

        num_positive_examples = min(
            len(negative_examples) / num_of_negatives_per_positive,
            len(positive_examples) - num_crowd_positives)

        positive_examples = sample(positive_examples,
                                   num_positive_examples + num_crowd_positives)

        crowd_positive_examples = positive_examples[0:num_crowd_positives]
        corpus_positive_examples = positive_examples[
            num_crowd_positives:len(positive_examples) - num_test_examples]
        testing_positive_examples = positive_examples[
            len(positive_examples) - num_test_examples:]

    #number_of_negatives_per_positive = (1.0 * len(negative_examples)) / (num_positive_examples - num_crowd_positives)

    print "Number of negative examples per positive example"
    print num_of_negatives_per_positive

    #Make all the files
    dataset_id = uuid.uuid1()
    
    filename_positive_crowd_examples = (
        'temp_datasets/%s_positives_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))
    filename_negative_crowd_examples = (
        'temp_datasets/%s_negatives_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))
    
    filename_unlabeled_corpus = (
        'temp_datasets/%s_corpus_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))
    filename_labeled_corpus = (
        'temp_datasets/%s_labeled_corpus_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))

    filename_positive_testing_examples = (
        'temp_datasets/%s_pos_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))
    filename_negative_testing_examples = (
        'temp_datasets/%s_neg_%d_%s' % (
            interested_category, num_of_negatives_per_positive, dataset_id))

    output_file_positive_crowd_examples = open(
        filename_positive_crowd_examples,'w')
    output_file_negative_crowd_examples = open(
        filename_negative_crowd_examples, 'w')
    
    output_file_unlabeled_corpus = open(
        filename_unlabeled_corpus, 'w')
    output_file_labeled_corpus = open(
        filename_labeled_corpus, 'w')

    output_file_positive_testing_examples = open(
        filename_positive_testing_examples, 'w')
    output_file_negative_testing_examples = open(
        filename_negative_testing_examples, 'w')

    #First make the file with the crowd generated examples
    for ex in crowd_positive_examples:
        output_file_positive_crowd_examples.write('%s\n' % ex)

    #Now make the unlabeled corpus
    for ex in corpus_positive_examples:
        output_file_unlabeled_corpus.write('%s\n' % ex)
        output_file_labeled_corpus.write('%s\t1\n' % ex)

    for i in range(int(num_of_negatives_per_positive *
                        len(corpus_positive_examples))):
        next_negative_example = negative_examples.pop()
        output_file_unlabeled_corpus.write('%s\n' %
                                           next_negative_example)
        output_file_labeled_corpus.write('%s\t0\n' %
                                         next_negative_example)

    #Now make the test set
    for ex in testing_positive_examples:
        output_file_positive_testing_examples.write('%s\n' % ex)

    for i in range(int(num_of_negatives_per_positive *
                        len(testing_positive_examples))):
        next_negative_example = negative_examples.pop()
        output_file_negative_testing_examples.write('%s\n' % 
                                                    next_negative_example)



    output_file_positive_crowd_examples.close()
    output_file_negative_crowd_examples.close()
    output_file_unlabeled_corpus.close()
    output_file_labeled_corpus.close()
    output_file_positive_testing_examples.close()
    output_file_negative_testing_examples.close()


    s3 = S3Connection(app.config['AWS_ACCESS_KEY_ID'],
                      app.config['AWS_SECRET_ACCESS_KEY'])

    bucket = s3.get_bucket("extremest-extraction-data-for-simulation")

    s3_key = bucket.new_key(filename_positive_crowd_examples)
    s3_key.set_contents_from_filename(filename_positive_crowd_examples)
    s3_key.make_public()
    positive_crowd_examples_url = s3_key.generate_url(3600000)

    s3_key = bucket.new_key(filename_negative_crowd_examples)
    s3_key.set_contents_from_filename(filename_negative_crowd_examples)
    s3_key.make_public()
    negative_crowd_examples_url = s3_key.generate_url(3600000)

    s3_key = bucket.new_key(filename_unlabeled_corpus)
    s3_key.set_contents_from_filename(filename_unlabeled_corpus)
    s3_key.make_public()
    unlabeled_corpus_url = s3_key.generate_url(3600000)

    s3_key = bucket.new_key(filename_labeled_corpus)
    s3_key.set_contents_from_filename(filename_labeled_corpus)
    s3_key.make_public()
    labeled_corpus_url = s3_key.generate_url(3600000)


    s3_key = bucket.new_key(filename_positive_testing_examples)
    s3_key.set_contents_from_filename(filename_positive_testing_examples)
    s3_key.make_public()
    positive_testing_examples_url = s3_key.generate_url(3600000)

    s3_key = bucket.new_key(filename_negative_testing_examples)
    s3_key.set_contents_from_filename(filename_negative_testing_examples)
    s3_key.make_public()
    negative_testing_examples_url = s3_key.generate_url(3600000)

    
    #Delete all the temp files
    os.remove(filename_positive_crowd_examples)
    os.remove(filename_negative_crowd_examples)
    os.remove(filename_unlabeled_corpus)
    os.remove(filename_labeled_corpus)
    os.remove(filename_positive_testing_examples)
    os.remove(filename_negative_testing_examples)
    
    
    return [positive_crowd_examples_url, negative_crowd_examples_url,
            unlabeled_corpus_url, labeled_corpus_url,
            positive_testing_examples_url, negative_testing_examples_url]

