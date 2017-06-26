import nltk
import pickle
import sys
import json
import requests
from app import app
from schema.experiment import Experiment
from schema.job import Job
from schema.gold_extractor import Gold_Extractor
from crowdjs_util import get_answers, get_questions, get_answers_for_question, get_task_data
from s3_util import insert_model_into_s3, insert_crf_model_into_s3
from ml.extractors.cnn_core.train import train_cnn
from ml.extractors.cnn_core.test import test_cnn
from ml.extractors.lr.lr import train_lr, test_lr
from ml.extractors.cnn_core.computeScores import computeScores
from random import sample, shuffle
from math import floor, ceil
import urllib2
import uuid
import shutil
import os
import subprocess
import re
import traceback
import time
import cPickle
from Queue import PriorityQueue
import heapq

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

    while True:
        try:
            r = requests.post(put_task_data_url,
                              json=data)
            r.raise_for_status()
            break
        except Exception:
            print "Exception while communicating with S3:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            sys.stdout.flush()
            time.sleep(10)
            continue

    print "Here is the response after trying to modify the task data"
    print r.text
    sys.stdout.flush()

    #return taboo_words
    

#Writes the model to a temporary file.
#The purpose of this is so tensorflow can read the model
def write_model_to_file(job_id = None, gold_extractor = None):

    temp_model_file_name = 'temp_models/%s' % str(uuid.uuid1())


    if not job_id == None:
        job = Job.objects.get(id = job_id)

        #model_file = urllib2.urlopen(job.model_file)
        #model_meta_file =  urllib2.urlopen(job.model_meta_file)
        while True:
            try:
                model_file = requests.get(job.model_file)
                model_file.raise_for_status()
                model_meta_file =  requests.get(job.model_meta_file)
                model_meta_file.raise_for_status()
                break
            except Exception:
                print "Exception while communicating with S3:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

    
        temp_model_file_handle = open(temp_model_file_name, 'wb')
        temp_model_file_handle.write(str(model_file.content))
        temp_model_file_handle.close()
        
        
        temp_model_meta_file_handle = open('%s.meta' % temp_model_file_name,
                                           'wb')
        temp_model_meta_file_handle.write(str(model_meta_file.content))
        temp_model_meta_file_handle.close()
    elif not gold_extractor == None:
        gold_extractor = Gold_Extractor.objects.get(name = gold_extractor)
        
        temp_model_file_handle = open(temp_model_file_name, 'wb')
        temp_model_file_handle.write(gold_extractor.model_file.read())
        temp_model_file_handle.close()
        
        
        temp_model_meta_file_handle = open('%s.meta' % temp_model_file_name,
                                           'wb')
        temp_model_meta_file_handle.write(gold_extractor.model_meta_file.read())
        temp_model_meta_file_handle.close()
        
    

    return temp_model_file_name


#Writes the model to a temporary file.
#The purpose of this is so tensorflow can read the model
def write_crf_model_to_file(job_id = None, gold_extractor = None):

    model_folder = str(uuid.uuid1())
    shutil.copytree('api/ml/extractors/crf', 
                    'api/ml/extractors/temp_extractors/%s' % model_folder)

    if not job_id == None:
        job = Job.objects.get(id = job_id)

        while True:
            try:
                model_file = requests.get(job.model_file)
                model_file.raise_for_status()
                break
            except Exception:
                print "Exception while communicating with crowdjs:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

        temp_model_file_handle = open(
            "api/ml/extractors/temp_extractors/%s/model.out" % model_folder, 
            'wb')
        temp_model_file_handle.write(str(model_file.content))
        temp_model_file_handle.close()
                

    elif not gold_extractor == None:
        gold_extractor = Gold_Extractor.objects.get(name = gold_extractor)
        
        temp_model_file_handle = open(
           "api/ml/extractors/temp_extractors/%s/model.out" % model_folder, 
            'wb')
        temp_model_file_handle.write(gold_extractor.model_file.read())
        temp_model_file_handle.close()
        
    
    return model_folder




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



def get_cost_of_action(category_id):

    category = app.config['EXAMPLE_CATEGORIES'][category_id]

    if category_id == 0:
        return app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * category['price']
    elif category_id == 1:
        return app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE']* category['price']
    elif category_id == 2:
        return app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * app.config[
            'CONTROLLER_LABELS_PER_QUESTION'] * category['price']



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
    #print "Getting answers from crowd_js for task %s" % task_id
    #print len(answers)
    #print "Expected number of answers"
    #print wait_until_batch_finished
    #sys.stdout.flush()

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
        question_ids = []
        for question in questions:
            question_id = question['_id']['$oid']
            question_ids.append(question_id)
            #print question_id
            #sys.stdout.flush()

        all_answers = get_answers_for_question(question_ids)
        
        answers_by_question_id = {}

        #print all_answers
        #sys.stdout.flush()

        for answer in all_answers:
            #print answer
            #sys.stdout.flush()
            answer_question_id = answer['question']['$oid']
            if not answer_question_id in answers_by_question_id:
                answers_by_question_id[answer_question_id] = []
            answers_by_question_id[answer_question_id].append(answer)
        

        for question in questions:
            question_data = question['data'].split('\t')
            question_id = question['_id']['$oid']
            sentence = question_data[len(question_data)-1]
            sentence = sentence.strip()
            answers = answers_by_question_id[question_id]
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
    
    #print "Finished parsing the examples and the labels"
    #sys.stdout.flush()

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
        print "Training a classiifer"
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
    

    if app.config['MODEL'] == 'LR':
        classifier, vocabulary = train_lr(
            training_positive_examples + training_negative_examples,
            ([1 for e in training_positive_examples] +
             [0 for e in training_negative_examples]))

        job = Job.objects.get(id=job_id)
        job.vocabulary = pickle.dumps(vocabulary)
        job.model_file = pickle.dumps(classifier)
        job.save()

        print "Model saved"
        sys.stdout.flush()

        return True
        
    if app.config['MODEL'] == 'CNN':
        job = Job.objects.get(id = job_id)
        
        model_file_name, vocabulary = train_cnn(
            training_positive_examples + training_negative_examples,
            ([1 for e in training_positive_examples] +
             [0 for e in training_negative_examples]),
            job.gpu_device_string)

        print "Saving the model"
        print job_id
        sys.stdout.flush()

        (model_url, model_meta_url, 
         model_key, model_meta_key) = insert_model_into_s3(
            model_file_name,
            "{}.meta".format(model_file_name))
        job = Job.objects.get(id=job_id)
        job.vocabulary = pickle.dumps(vocabulary)
    
    
        job.model_file = model_url
        job.model_meta_file = model_meta_url
    
        print "Model saved"
        sys.stdout.flush()
        
        job.num_training_examples_in_model = (
            len(training_positive_examples) + len(training_negative_examples))

        print "training saved"
        sys.stdout.flush()
        
        job.save()

        model_folder = re.search('\/[^/]+-[^/]+-[^/]+-[^/]+-[^/]+\/', 
                                 model_file_name).group(0)
        model_folder = model_folder.split('/')[1]
        shutil.rmtree('runs/%s' % model_folder )
                        
        print "Job modified"        
        print "file handles closed"
        sys.stdout.flush()
    
        return True
    
    elif app.config['MODEL'] == 'CRF':
        model_folder = str(uuid.uuid1())
        shutil.copytree('api/ml/extractors/crf', 
                        'api/ml/extractors/temp_extractors/%s' % model_folder)
        
        training_data_directory = os.path.join(
            os.getcwd(),
            'api/ml/extractors/temp_extractors/%s/training_data' % 
            model_folder)

        os.mkdir(training_data_directory)

        print "Writing Training Data"        
        print model_folder
        print training_data_directory
        print os.path.exists(training_data_directory)
        sys.stdout.flush()    

        training_data_file = open(
            'api/ml/extractors/temp_extractors/%s/training_data/training_data'%
            model_folder,
            'w')
        
        for training_positive_example in training_positive_examples:
            training_data_file.write(
                'event\t%s\n' % training_positive_example.encode(
                    'utf8').replace('\n', ''))
        for training_negative_example in training_negative_examples:
            training_data_file.write(
                'NO_EVENT\t%s\n' % training_negative_example.encode(
                    'utf8').replace('\n',''))
        
        training_data_file.close()

        print "Done Writing Training Data"        
        print model_folder
        sys.stdout.flush()

        train_process = subprocess.Popen(
            [
            os.path.join(
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s/train.sh" %model_folder),
            os.path.join(
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s/training_data" % 
                model_folder)
            ],
            cwd=os.path.join(
                #os.path.abspath(sys.path[0]), 
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s" % model_folder))
        train_process.wait() 
        

        shutil.copyfile(
            "api/ml/extractors/temp_extractors/%s/model.out" % model_folder,
            "api/ml/extractors/temp_extractors/%s/%s" % (model_folder, 
                                                         model_folder))
        model_url = insert_crf_model_into_s3(
            "api/ml/extractors/temp_extractors/%s/%s" % (model_folder,
                                                         model_folder))

        job.model_file = model_url
        job.save()

        shutil.rmtree('api/ml/extractors/temp_extractors/%s' % 
                      model_folder)
                
        return True

def test(job_id, test_examples, test_labels):
    job = Job.objects.get(id = job_id)

    if app.config['MODEL'] == 'LR':
        vocabulary = pickle.loads(job.vocabulary)
        model = pickle.loads(job.model_file)
        predicted_labels, label_probabilities = test_lr(
            test_examples, test_labels, model, vocabulary)

        return predicted_labels, label_probabilities
    if app.config['MODEL'] == 'CNN':
        vocabulary = pickle.loads(job.vocabulary)
        temp_file_name = write_model_to_file(job_id)
        predicted_labels, label_probabilities = test_cnn(
            test_examples, test_labels,
            temp_file_name,
            vocabulary)
        os.remove(os.path.join(
            os.getcwd(),temp_file_name))
        os.remove(os.path.join(
            os.getcwd(),'%s.meta' % temp_file_name))

        return predicted_labels, label_probabilities
    elif app.config['MODEL'] == 'CRF':
        model_folder = write_crf_model_to_file(job_id)

        os.makedirs(
            os.path.join(
                os.getcwd(),
                'api/ml/extractors/temp_extractors/%s/testing_data' % 
                model_folder))

        testing_data_file = open(
            'api/ml/extractors/temp_extractors/%s/testing_data/testing_data' %
            model_folder,
            'w')

        for test_example, test_label in zip(test_examples, test_labels):
            #if test_label == 0:
            #    testing_data_file.write(
            #        '%s\n' % test_example)
            #elif test_label == 1:
            #    testing_data_file.write(
            #        '%s\n' % test_example)
            testing_data_file.write(
                '%s\n' % test_example)


        test_process = subprocess.Popen(
            [
            os.path.join(
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s/run.sh" %model_folder),
            os.path.join(
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s/testing_data" % 
                model_folder)
            ],
            cwd=os.path.join(
                #os.path.abspath(sys.path[0]), 
                os.getcwd(),
                "api/ml/extractors/temp_extractors/%s" % model_folder))
        test_process.wait()                                 


        predicted_labels = []
        predicted_labels_file = open(
            'api/ml/extractors/temp_extractors/%s/output' % model_folder, 'r')

        for predicted_label in predicted_labels_file:
            predicted_label = predicted_label.split('\t')[0]
            if predicted_label == "NO_EVENT":
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)


        #shutil.rmtree('api/ml/extractors/temp_extractors/%s' % 
        #              model_folder)

        print "PREDICTED LABELS"
        print predicted_labels
        sys.stdout.flush()

        return predicted_labels, []




def get_gold_labels(job, selected_examples):

    if not 'https' in job.gold_extractor:
        gold_extractor = Gold_Extractor.objects.get(
            name=job.gold_extractor)
        model_file_name = write_model_to_file(
            gold_extractor = gold_extractor.name)
        vocabulary = cPickle.loads(str(gold_extractor.vocabulary))
        predicted_labels, label_probabilities = test_cnn(
            selected_examples,
            [0 for i in selected_examples],
            model_file_name,
            vocabulary)

        os.remove(os.path.join(
            os.getcwd(), model_file_name))
        os.remove(os.path.join(
            os.getcwd(),'%s.meta' % model_file_name))
    else:
        gold_labels = {}

        while True:
            try:
                r = requests.get(job.gold_extractor)
                r.raise_for_status()
                gold_corpus = str(r.content).split('\n')
                break
            except Exception:
                print "Exception while communicating with S3:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

        for line in gold_corpus:
            if line == "":
                continue

            line = line.split('\t')
            example = line[0]
            #example = unicode(line[0], 'utf-8')
            label = int(line[1])
            gold_labels[example] = label

        predicted_labels = []
        for example in selected_examples:
            predicted_labels.append(gold_labels[example])
    
    return predicted_labels




######
#Gets examples that are predicted positive
######
def get_unlabeled_examples_from_corpus(task_ids, task_categories,
                                       training_examples, training_labels,
                                       task_information, costSoFar,
                                       budget, job_id):
    
    print "choosing to find examples from corpus and label them"
    sys.stdout.flush()
    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    job = Job.objects.get(id=job_id)

    #num_positive_examples_to_label = int(
    #    app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 2.0)
    num_positive_examples_to_label = app.config[
        'CONTROLLER_LABELING_BATCH_SIZE']

    num_negative_examples_to_label = (
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] -
        num_positive_examples_to_label)
    

    training_positive_examples = []
    training_negative_examples = []
    for training_example_set, training_label_set in zip(
            training_examples, training_labels):
        #print "TRAINING EXAMPLE SET"
        #print training_example_set
        #print training_label_set
        #sys.stdout.flush()

        for (training_example,
             training_label) in zip(training_example_set,
                                    training_label_set):
            if training_label == 1:
                training_positive_examples.append(training_example)
            elif training_label == 0:
                training_negative_examples.append(training_example)
            else:
                print "This condition should not have happened"
                sys.stdout.flush()
                raise Exception

    #print training_positive_examples
    #print training_negative_examples
    #sys.stdout.flush()

    retrain(job_id, ['all'], [],
            training_positive_examples, 
            training_negative_examples)

    test_examples = []
    test_labels = []

    while True:
        try:
            r = requests.get(job.unlabeled_corpus)
            r.raise_for_status()
            corpus = str(r.content).split('\n')
            break
        except Exception:
                print "Exception while communicating with S3:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

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

    corpus = set(corpus)-set(used_examples)
    for sentence in corpus:
        if sentence == "":
            continue
        test_examples.append(sentence)
        test_labels.append(0)



    job = Job.objects.get(id = job_id)

    predicted_labels, label_probabilities = test(
        job_id,
        test_examples,
        test_labels)


    positive_examples = []
    negative_examples = []
    for i in range(len(predicted_labels)):
        predicted_label = predicted_labels[i]
        example = test_examples[i]
        if predicted_label == 1:
            positive_examples.append(example)
        elif predicted_label == 0:
            negative_examples.append(example)
        else:
            print "This should not happen"
            raise Exception


    print "Sampling examples from the corpus"
    sys.stdout.flush()

    selected_examples = []
    expected_labels = []
    if len(positive_examples) < num_positive_examples_to_label:
        selected_examples += positive_examples
        expected_labels += [1 for i in range(len(positive_examples))]
        selected_examples += sample(
            negative_examples,
            app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(positive_examples))
        expected_labels += [0 for i in range(
                app.config['CONTROLLER_LABELING_BATCH_SIZE']-
            len(positive_examples))]
    elif len(negative_examples) < num_negative_examples_to_label:
        selected_examples += negative_examples
        expected_labels += [0 for i in range(len(negative_examples))]
        selected_examples += sample(
            positive_examples,
            app.config['CONTROLLER_LABELING_BATCH_SIZE']-len(negative_examples))
        expected_labels += [1 for i in range(
            app.config['CONTROLLER_LABELING_BATCH_SIZE']-
            len(negative_examples))]
    else:
        selected_examples += sample(positive_examples,
                                    num_positive_examples_to_label)
        expected_labels += [1 for i in range(num_positive_examples_to_label)]
        selected_examples += sample(negative_examples,
                                    num_negative_examples_to_label)
        expected_labels += [0 for i in range(num_negative_examples_to_label)]

    print "Shuffling examples from the corpus"
    sys.stdout.flush()

    shuffle(selected_examples)

    return selected_examples, expected_labels


#Gets random examples
def get_random_unlabeled_examples_from_corpus(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id):
    
    print "choosing to find examples from corpus and label them"
    sys.stdout.flush()
    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    job = Job.objects.get(id = job_id)
    test_examples = []

    while True:
        try:
            r = requests.get(job.unlabeled_corpus)
            r.raise_for_status()
            corpus = str(r.content).split('\n')
            break
        except Exception:
                print "Exception while communicating with S3:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

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


    corpus = set(corpus)-set(used_examples)
    for sentence in corpus:
        if sentence == "":
            continue
        test_examples.append(sentence)

    print "Sampling from a corpus of size"
    print len(test_examples)
    sys.stdout.flush()

    selected_examples = sample(test_examples,
                                app.config['CONTROLLER_LABELING_BATCH_SIZE'])
    expected_labels = [0 for i in range(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'])]
        

    return selected_examples, expected_labels




#Gets examples by uncertainty sampling
def get_US_unlabeled_examples_from_corpus(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id):
    
    print "choosing to find examples from corpus and label them"
    sys.stdout.flush()
    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    job = Job.objects.get(id = job_id)
    test_examples = []

    while True:
        try:
            r = requests.get(job.unlabeled_corpus)
            r.raise_for_status()
            corpus = str(r.content).split('\n')
            break
        except Exception:
                print "Exception while communicating with S3:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
                sys.stdout.flush()
                time.sleep(10)
                continue

    #Get all the previous examples that we labeled already and do not 
    #include them.
    used_examples = []
    for i, task_category in zip(range(len(task_categories)), task_categories):
        #This check is because some data in the database is inconsistent
        if isinstance(task_category, dict):
            task_category_id = task_category['id']
        else:
            task_category_id = task_category
        if task_category_id == 2:
            used_examples += training_examples[i]


    corpus = set(corpus)-set(used_examples)
    for sentence in corpus:
        if sentence == "":
            continue
        test_examples.append(sentence)

    print "Sampling from a corpus of size"
    print len(test_examples)
    sys.stdout.flush()


    training_positive_examples = []
    training_negative_examples = []
    for training_example_set, training_label_set in zip(training_examples,
                                                        training_labels):
        for training_example, training_label in zip(training_example_set,
                                                    training_label_set):
            if training_label == 1:
                training_positive_examples.append(training_example)
            elif training_label == 0:
                training_negative_examples.append(training_example)
            else:
                raise Exception


    retrain(job_id, ['all'], [],
            training_positive_examples, 
            training_negative_examples)

    predicted_labels, label_probabilities = test(
        job_id,
        test_examples,
        [0 for ex in test_examples])


    pq = [-2.0 for i in range(app.config['CONTROLLER_LABELING_BATCH_SIZE'])]
    heapq.heapify(pq)

    
    for test_example, label_probability in zip(test_examples, 
                                               label_probabilities):
        #print "EXAMPLE BEING INSERTED"
        #print max(label_probability)
        #sys.stdout.flush()

        heapq.heappushpop(pq, (-1.0 * max(label_probability), test_example))


    #examples_in_decreasing_uncertainty = []
    #for i in range(app.config['CONTROLLER_LABELING_BATCH_SIZE']):
    #    examples_in_decreasing_uncertainty.append(pq.get()[1])

    #examples_in_decreasing_uncertainty = sorted(
    #    zip(test_examples, label_probabilities),
    #    key=lambda x: max(x[1]))
    

    print "HERE ARE THE EXAMPLES IN DECREASING UNCERTAINTY"
    print heapq.nlargest(app.config['CONTROLLER_LABELING_BATCH_SIZE'], pq)
    sys.stdout.flush()


    expected_labels = [0 for i in range(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'])]
        

    #Get a batch
    #most_uncertain_examples = [
    #    ex for (ex, probs) in 
    #examples_in_decreasing_uncertainty[
    #        0:app.config['CONTROLLER_LABELING_BATCH_SIZE']]]

    
    return  ([ex for (p, ex) in heapq.nlargest(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'], pq)],
            expected_labels)

#Gets examples predicted as positive with ties broken by uncertainty sampling
def get_US_PP_unlabeled_examples_from_corpus(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id):
    
    print "applying uncertainty sampling to predicted positives"
    sys.stdout.flush()
 
    (selected_examples,
     expected_labels) = get_unlabeled_examples_from_corpus(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id)


    predicted_labels, label_probabilities = test(
        job_id,
        selected_examples,
        [0 for ex in selected_examples])


    pq = [-2.0 for i in range(app.config['CONTROLLER_LABELING_BATCH_SIZE'])]
    heapq.heapify(pq)

    
    for selected_example, label_probability in zip(selected_examples, 
                                               label_probabilities):

        heapq.heappushpop(pq, (-1.0 * max(label_probability), 
                               selected_example))

    

    #print "HERE ARE THE EXAMPLES IN DECREASING UNCERTAINTY"
    #print heapq.nlargest(app.config['CONTROLLER_LABELING_BATCH_SIZE'], pq)
    #sys.stdout.flush()


    expected_labels = [0 for i in range(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'])]
        

    #Get a batch
    #most_uncertain_examples = [
    #    ex for (ex, probs) in 
    #examples_in_decreasing_uncertainty[
    #        0:app.config['CONTROLLER_LABELING_BATCH_SIZE']]]

    
    return  ([ex for (p, ex) in heapq.nlargest(
        app.config['CONTROLLER_LABELING_BATCH_SIZE'], pq)],
            expected_labels)
