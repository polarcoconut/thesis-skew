import nltk
import pickle
import sys
import json
import requests
from app import app
from schema.job import Job
from crowdjs_util import get_answers, get_questions, get_answers_for_question
from ml.extractors.cnn_core.train import train_cnn


#old_taboo_words is a python pickle that is actually a dictionary
#mapping words to the number of times
#they have been used
@app.celery.task(name='compute_taboo_words')
def compute_taboo_words(old_taboo_words, old_sentence, new_sentence, task_id,
                        requester_id, put_task_data_url):
    nltk.download('punkt')
    nltk.download('stopwords')

    old_taboo_words = pickle.loads(old_taboo_words)
                
    print "FINDING TABOO WORDS"
    sys.stdout.flush()
    
    #Find the taboo words
    tokenized_old_sentence = nltk.word_tokenize(old_sentence.lower())
    tokenized_new_sentence = nltk.word_tokenize(new_sentence.lower())
    
    new_taboo_words = set(tokenized_new_sentence) - set(tokenized_old_sentence)
    new_taboo_words = new_taboo_words - set(nltk.corpus.stopwords.words('english'))

    #Add the new taboo words to the existing taboo words
    #and only add it if it's greater than or equal to 3 characters.
    for new_taboo_word in new_taboo_words:
        if len(new_taboo_word) < 3:
            continue
        if not new_taboo_word in old_taboo_words:
            old_taboo_words[new_taboo_word] = 1
        else:
            old_taboo_words[new_taboo_word] += 1

    #Return a pickled new taboo dictionary.

    print "Here is the new dictionary of taboo words"
    print old_taboo_words
    sys.stdout.flush()
    print "Posting to"
    print put_task_data_url
    sys.stdout.flush()


    #convert to string
    taboo_words = pickle.dumps(old_taboo_words)


    #Put the new task data into crowdjs.
    #headers = {'Authentication-Token': app.CROWDJS_API_KEY,
    #           'content_type' : 'application/json'}
    
    data = {'task_id' : task_id ,
            'requester_id' : requester_id,
            'data' : taboo_words}

    r = requests.post(put_task_data_url,
                      json=data)

    print "Here is the response after trying to modify the task data"
    print r.text
    sys.stdout.flush()

    return taboo_words
    


def write_model_to_file(job_id):
    job = Job.objects.get(id = job_id)
    
    temp_model_file_handle = open('temp_model_file', 'wb')
    temp_model_file_handle.write(job.model_file)
    temp_model_file_handle.close()
    
    
    temp_model_meta_file_handle = open('temp_model_file.meta', 'wb')
    temp_model_meta_file_handle.write(job.model_meta_file)
    temp_model_meta_file_handle.close()


    return 'temp_model_file'




def parse_task_information(args):
    event_name = args['event_name']
    event_definition = args['event_definition']
    event_pos_example_1 = args['event_pos_example_1']
    event_pos_example_1_trigger = args['event_pos_example_1_trigger']
    event_pos_example_2 = args['event_pos_example_2']
    event_pos_example_2_trigger = args['event_pos_example_2_trigger']
    event_pos_example_nearmiss = args['event_pos_example_nearmiss']
    
    event_neg_example = args['event_neg_example']
    event_neg_example_nearmiss = args['event_neg_example_nearmiss']
    

    task_information = (event_name, event_definition,
                        event_pos_example_1,
                        event_pos_example_1_trigger,
                        event_pos_example_2,
                        event_pos_example_2_trigger,
                        event_pos_example_nearmiss,
                        event_neg_example,
                        event_neg_example_nearmiss)

    
    return task_information



def getLatestCheckpoint(job_id):
    #read the latest checkpoint
    job = Job.objects.get(id = job_id)

    print "GETTING LATEST CHECKPOINT"
    print job_id
    
    timestamps = job.checkpoints.keys()
        
    print timestamps
    sys.stdout.flush()
    
    
    most_recent_timestamp = max([int(x) for x in timestamps])

    checkpoint = job.checkpoints[str(most_recent_timestamp)]
    (task_information, budget) = pickle.loads(job.task_information)

    return (task_information, budget, checkpoint)

def split_examples(task_ids, task_categories, positive_types = [],
                   only_sentence=True):
    positive_examples = []
    negative_examples = []
    for task_id, task_category_id in zip(task_ids,task_categories):
        answers = parse_answers(task_id, task_category_id,
                                False, positive_types, only_sentence)
        print answers
        new_examples, new_labels = answers
        for new_example, new_label in zip(new_examples, new_labels):
            if new_label == 1:
                positive_examples.append(new_example)
            else:
                negative_examples.append(new_example)
    print positive_examples
    print negative_examples
    return positive_examples, negative_examples

#because of old data structures, category_id might be a category structure
#
#  only_sentence : return only the sentence, not all the other crowdsourced
#                  details
#
def parse_answers(task_id, category_id, wait_until_batch_finished= -1,
                  positive_types = [], only_sentence = True):


    if not isinstance(category_id, dict):        
        category = app.config['EXAMPLE_CATEGORIES'][category_id]
    else:
        category = category_id

            
    answers = get_answers(task_id)
    print "Getting answers from crowd_js for task %s" % task_id
    print len(answers)
    print "Expected number of answers"
    print wait_until_batch_finished
    sys.stdout.flush()
    
    if (len(answers) < wait_until_batch_finished):
        return None

    examples = []
    labels = []
    
    #Data structure of checkpoints were updated so old checkpoints may not have
    #an id in them.
    if (('id' in category and category['id'] == 0) or
        category['task_name'] == 'Event Generation'):
        for answer in answers:
            value = answer['value']
                
            value = value.split('\t')
            sentence = value[0]
            trigger = value[1]
            
            if len(value) > 2:
                if value[2] == 'Yes':                
                    past = True
                else:
                    past = False
                    
                if value[3] == 'Yes':
                    future = True
                else:
                    future = False
                    
                if value[4] == 'Yes':
                    general = True
                else:
                    general = False
            
            
                if 'all' in positive_types:
                    labels.append(1)
                elif 'past' in positive_types and past:
                        labels.append(1)                
                elif 'future' in positive_types and future:
                        labels.append(1)
                elif 'general' in positive_types and general:
                        labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(1)


            if only_sentence:
                examples.append(sentence)
            else:
                examples.append(answer['value'])

    elif (('id' in category and category['id'] == 1) or
          category['task_name'] == 'Event Negation'):
        for answer in answers:
            value = answer['value']
            value = value.split('\t')
            sentence = value[0]
            previous_sentence_not_example_of_event = value[1]

            if len(value) > 2:
                if value[2] == 'failing':                
                    failing = True
                else:
                    failing = False

                if 'all' in positive_types:
                    labels.append(0)
                elif 'general' in positive_types:
                    if failing:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            else:
                labels.append(0)

            if only_sentence:
                examples.append(sentence)
            else:
                examples.append(answer['value'])
    elif (('id' in category and category['id'] == 2) or
          category['task_name'] == 'Event Labeling'):
        questions = get_questions(task_id)
        for question in questions:
            print question
            sys.stdout.flush()
            question_id = question['_id']['$oid']
            question_data = question['data'].split('\t')
            sentence = question_data[len(question_data)-1]
            sentence = sentence.strip()
            answers = get_answers_for_question(question_id)

            label = 0
            past = 0
            future = 0
            general = 0
            hypothetical = 0
            
            for answer in answers:
                print answer
                sys.stdout.flush()

                if 'value' not in answer:
                    continue
                
                value = answer['value']
                value = value.split('\t')
                
                if len(value) > 2:
                    label += 1
                    if value[2] == 'Yes':
                        past += 1
                    else:
                        past -= 1
                    if value[3] == 'Yes':
                        future += 1
                    else:
                        future -= 1
                    if value[4] == 'Yes':
                        general += 1
                    else:
                        general -= 1
                    if value[5] == 'Yes':
                        hypothetical += 1
                    else:
                        hypothetical -= 1
                else:
                    label -= 1
                    if value[1] == 'Yes':
                        hypothetical += 1
                    else:
                        hypothetical -= 1
                    
            if only_sentence:
                examples.append(sentence)
            else:
                if past > 0:
                    sentence = sentence + '\tPAST(%d)' % past
                if future > 0:
                    sentence = sentence + '\tFUTURE(%d)' % future
                if general > 0:
                    sentence = sentence + '\tGENERAL(%d)' % general
                if hypothetical > 0:
                    sentence = sentence + '\tHYPOTHETICAL(%d)' % hypothetical
                examples.append(sentence)

            if label > 0:
                labels.append(1)
            else:
                labels.append(0)

    return examples, labels

#Trains a CNN
@app.celery.task(name='retrain')
def retrain(job_id, positive_types):
    print "Training a CNN"
    sys.stdout.flush()
    task_information, budget, checkpoint = getLatestCheckpoint(job_id)
    (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)


    #Collect all the task_ids, leaving some out for the purposes of
    #cross validation
    #training_task_ids = []
    #training_task_categories = []
    
    
    training_positive_examples, training_negative_examples = split_examples(
        task_ids[2:],
        task_categories[2:],
        positive_types)
    
    model_file_name, vocabulary = train_cnn(
        training_positive_examples + training_negative_examples,
        ([1 for e in training_positive_examples] +
         [0 for e in training_negative_examples]))


    model_file_handle = open(model_file_name, 'rb')
    model_binary = model_file_handle.read()

    model_meta_file_handle = open("{}.meta".format(model_file_name), 'rb')
    model_meta_binary = model_meta_file_handle.read()

    print "Saving the model"
    print job_id
    sys.stdout.flush()

    job = Job.objects.get(id=job_id)
    job.vocabulary = pickle.dumps(vocabulary)
    job.model_file = model_binary
    job.model_meta_file = model_meta_binary
    print "Model saved"
    sys.stdout.flush()

    job.num_training_examples_in_model = (
        len(training_positive_examples) + len(training_negative_examples))

    print "training saved"
    sys.stdout.flush()

    job.save()

    print "Job modified"
    sys.stdout.flush()

    model_file_handle.close()
    model_meta_file_handle.close()

    print "file handles closed"
    sys.stdout.flush()
    
    return True
