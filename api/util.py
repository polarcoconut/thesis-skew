import nltk
import pickle
import sys
import json
import requests
from app import app
from schema.job import Job
from crowdjs_util import get_answers, get_questions, get_answers_for_question, get_task_data
from ml.extractors.cnn_core.train import train_cnn
from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.cnn_core.computeScores import computeScores
from random import sample, shuffle
from math import floor, ceil
import urllib2


#old_taboo_words is a python pickle that is actually a dictionary
#mapping words to the number of times
#they have been used
@app.celery.task(name='compute_taboo_words',
                 queue='taboo-queue')
def compute_taboo_words(old_sentence, new_sentence, task_id,
                        requester_id, put_task_data_url):
    nltk.download('punkt')
    nltk.download('stopwords')


    old_taboo_words = pickle.loads(get_task_data(task_id))
                
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

    #return taboo_words
    

#Writes the model to a temporary file.
#The purpose of this is so tensorflow can read the model
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
            
    most_recent_timestamp = max([int(x) for x in timestamps])

    checkpoint = job.checkpoints[str(most_recent_timestamp)]

    return checkpoint

    #(task_information, budget) = pickle.loads(job.task_information)
    #return (task_information, budget, checkpoint)

def split_examples(task_ids, task_categories, positive_types = ['all'],
                   only_sentence=True):
    positive_examples = []
    negative_examples = []
    for task_id, task_category_id in zip(task_ids,task_categories):
        answers = parse_answers(task_id, task_category_id,-1,
                                positive_types, only_sentence)
        new_examples, new_labels = answers
        for new_example, new_label in zip(new_examples, new_labels):
            if new_label == 1:
                positive_examples.append(new_example)
            else:
                negative_examples.append(new_example)
    return positive_examples, negative_examples

#because of old data structures, category_id might be a category structure
#
#  only_sentence : return only the sentence, not all the other crowdsourced
#                  details
#
#  return: examples, labels
#
def parse_answers(task_id, category_id, wait_until_batch_finished= -1,
                  positive_types = ['all'], only_sentence = True):


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

    # If the number of answers we have is less than what we expect,
    # don't do anything. If we expect -1 or False, then
    # give us the answers.
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
            question_id = question['_id']['$oid']
            question_data = question['data'].split('\t')
            sentence = question_data[len(question_data)-1]
            sentence = sentence.strip()
            answers = get_answers_for_question(question_id)
            if len(answers) == 0:
                continue

            label = 0
            past = 0
            future = 0
            general = 0
            hypothetical = 0
            
            for answer in answers:

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
                sentence += '\tVOTES(%d/%d)' % (label, len(answers))
                if past > 0:
                    sentence += '\tPAST(%d/%d)' % (past, len(answers))
                if future > 0:
                    sentence += '\tFUTURE(%d/%d)' % (future, len(answers))
                if general > 0:
                    sentence += '\tGENERAL(%d/%d)' % (general, len(answers))
                if hypothetical > 0:
                    sentence += '\tHYPOTHETICAL(%d/%d)' % (hypothetical,
                                                           len(answers))
                examples.append(sentence)

            if label > 0:
                labels.append(1)
            else:
                labels.append(0)

    return examples, labels

#Trains a CNN
#
# If you provide the actual examples, it will train  using these examples
# If you provide a list of task_ids, it will use the examples from these tasks
# If you provide neither, it will use all available examples in the job
#    and leave out the first two tasks as a validation set.
#
@app.celery.task(name='retrain')
def retrain(job_id, positive_types, task_ids_to_train = [],
            training_positive_examples = [], training_negative_examples = []):

    if training_positive_examples == [] and training_negative_examples == []:
        print "Training a CNN"
        sys.stdout.flush()

        job = Job.objects.get(id = job_id)
        
        checkpoint = getLatestCheckpoint(job_id)
        (task_information, budget) = pickle.loads(job.task_information)
        
        (task_ids, task_categories, costSoFar) = pickle.loads(checkpoint)

        if task_ids_to_train == []:
            ####
            # The first two tasks are used for cross validation purposes
            ####
            #task_ids_to_train = task_ids[2:]
            #task_categories_to_train = task_categories[2:]
            task_ids_to_train = task_ids
            task_categories_to_train = task_categories

        else:
            task_categories_to_train = []
            for task_id, task_category in zip(task_ids, task_categories):
                if task_id in task_ids_to_train:
                    task_categories_to_train.append(task_category)
        
        training_positive_examples, training_negative_examples = split_examples(
            task_ids_to_train,
            task_categories_to_train,
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



def get_unlabeled_examples_from_tackbp(task_ids, task_categories,
                                       training_examples, training_labels,
                                       task_information, costSoFar,
                                       budget, job_id):
    
    print "choosing to find examples from TACKBP and label them"
    sys.stdout.flush()
    next_category = app.config['EXAMPLE_CATEGORIES'][2]


    num_positive_examples_to_label = int(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 2.0)
    num_negative_examples_to_label = (
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] -
        num_positive_examples_to_label)
    

    retrain(job_id, ['all'])

    test_examples = []
    test_labels = []

    tackbp_newswire_corpus = urllib2.urlopen(
        app.config['TACKBP_NW_09_CORPUS_URL'])

    #Get all the previous examples that we labeled already
    used_examples = []
    for i, task_category in zip(range(len(task_categories)), task_categories):
        #This check is because some data in the database is inconsistent
        if isinstance(task_category, dict):
            task_category_id = task_category['id']
        else:
            task_category_id = task_category
        if task_category_id == 2:
            used_examples += training_examples[i]

    tackbp_newswire_corpus = set(tackbp_newswire_corpus)-set(used_examples)
    for sentence in tackbp_newswire_corpus:
        test_examples.append(sentence)
        test_labels.append(0)



    job = Job.objects.get(id = job_id)
    vocabulary = pickle.loads(job.vocabulary)

    predicted_labels = test_cnn(
        test_examples,
        test_labels,
        write_model_to_file(job_id),
        vocabulary)


    positive_examples = []
    negative_examples = []
    for i in range(len(predicted_labels)):
        predicted_label = predicted_labels[i]
        example = test_examples[i]
        if predicted_label == 1:
            positive_examples.append(example)
        else:
            negative_examples.append(example)



    print "Sampling examples from the corpus"
    sys.stdout.flush()

    selected_examples = []
    expected_labels = []
    if positive_examples < num_positive_examples_to_label:
        selected_examples += positive_examples
        selected_examples += sample(
            negative_examples,
            app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(positive_examples))
    elif negative_examples < num_negative_examples_to_label:
        selected_examples += negative_examples
        selected_examples += sample(
            positive_examples,
            app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(negative_examples))
    else:
        selected_examples += sample(positive_examples,
                                    num_positive_examples_to_label)
        expected_labels.append(1)
        selected_examples += sample(negative_examples,
                                    num_negative_examples_to_label)
        expected_labels.append(0)

    print "Shuffling examples from the corpus"
    sys.stdout.flush()

    shuffle(selected_examples)

    return selected_examples, expected_labels
