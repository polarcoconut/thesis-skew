from app import app
import sys
from mturk_connection import MTurk_Connection
from api.crowdjs_util import get_next_assignment, submit_answer, submit_answers, get_answers
from api.util import parse_answers, write_model_to_file, test
from api.ml.extractors.cnn_core.test import test_cnn
from random import shuffle
import uuid
import pickle
import json
from schema.job import Job
from schema.experiment import Experiment
from schema.gold_extractor import Gold_Extractor
import cPickle
import urllib2
from random import sample, random
from api.s3_util import insert_connection_into_s3
import requests


class MTurk_Connection_Super_Sim(MTurk_Connection):


    def __init__(self, experiment_id, job_id):

        print "Initializing Simulated Turk"
        sys.stdout.flush()

        self.job_id = job_id

        experiment = Experiment.objects.get(id=experiment_id)
        job = Job.objects.get(id=job_id)

        files_for_simulation = pickle.loads(job.files_for_simulation)

        self.modify_data = {}
        self.generate_data = []

        

        # If the gold extractor is a cached neural net and our 
        # generated data is from the crowd:
        if not 'https' in job.gold_extractor:
            """
            generate_files = files_for_simulation[0]
            for generate_file in generate_files:
                with open(generate_file, 'r') as generate_file_handle:
                    for line in generate_file_handle:
                        example = json.loads(line)
                        #print example
                        #sys.stdout.flush()
                        example = example['value'].split('\t')[0]
                        #print example
                        #sys.stdout.flush()

                        self.generate_data.append(example)
                        self.modify_data[example] = []

            shuffle(self.generate_data)
            print self.generate_data[0:3]
            print "Done getting generate data ready."
            sys.stdout.flush()
            """
            modify_files = files_for_simulation[1]
            for modify_file in modify_files:
                modify_file_handle = urllib2.urlopen(modify_file)
                for line in modify_file_handle:
                    example = json.loads(line)
                    value = example['value'].split('\t')
                    sentence = value[0]
                    previous_sentence_not_example_of_event = value[1]
                    old_sentence = value[3]
                    if old_sentence not in self.modify_data:
                        self.modify_data[old_sentence] = []
                    self.modify_data[old_sentence].append(sentence)

            self.generate_data = self.modify_data.keys()
            shuffle(self.generate_data)

            print "Done getting generate and modify data ready"
            sys.stdout.flush()

            gold_extractor = Gold_Extractor.objects.get(
                name=job.gold_extractor)
            
            model_file_name = write_model_to_file(
                gold_extractor = gold_extractor.name)
            
            self.model_file_name = model_file_name
            self.model_meta_file_name = "{}.meta".format(model_file_name)
            self.vocabulary = cPickle.loads(str(gold_extractor.vocabulary))
            self.use_gold_extractor = True

        #Otherwise, if the "gold extractor" is just data
        else:
            
            generate_files = files_for_simulation[0]
            for generate_file in generate_files:
                generate_file_handle = urllib2.urlopen(generate_file)
                for line in generate_file_handle:
                    self.generate_data.append(line.rstrip())
                        
            shuffle(self.generate_data)
            #print self.generate_data[0:3]
            print "Done getting generate data ready."
            sys.stdout.flush()
            
            #modify_files = files_for_simulation[1]
            #for modify_file in modify_files:
            #    modify_file_handle = urllib2.urlopen(modify_file)
            #    for line in modify_file_handle:
            #        example = json.loads(line)
            #        value = example['value'].split('\t')
            #        sentence = value[0]
            #        previous_sentence_not_example_of_event = value[1]
            #        old_sentence = value[3]
            #        if old_sentence not in self.modify_data:
            #            self.modify_data[old_sentence] = []
            #        self.modify_data[old_sentence].append(sentence)


            #########
            #USE NEAR-MISSES
            #########
            #old_sentence = next_assignment_question_data[9]
            #modified_sentence = sample(
            #    self.modify_data[old_sentence], 1)[0]

            #answer = (modified_sentence + "\tNotPos\tHypOrGen\t" +
            #          old_sentence + "\tSimTaboo")


            ########
            #USE RANDOM NEGATIVES
            #########
            job = Job.objects.get(id=self.job_id)

            while True:
                try:
                    r = requests.get(job.unlabeled_corpus).content
                    break
                except Exception:
                    print "Exception while communicating with S3"
                    print '-'*60
                    traceback.print_exc(file=sys.stdout)
                    print '-'*60
                    sys.stdout.flush()
                    time.sleep(60)
                    continue

            self.unlabeled_corpus = str(r).split('\n')
            shuffle(self.unlabeled_corpus)

            #self.generate_data = self.modify_data.keys()
            #shuffle(self.generate_data)

            print "Done getting generate and modify data ready"
            sys.stdout.flush()


            #################
            # Make a way to figure out what the correct labels are so
            # that you can simulate workers.
            #################
            gold_labels = {}

            print  job.gold_extractor
            sys.stdout.flush()
            
            gold_corpus = str(requests.get(
                job.gold_extractor).content).split('\n')
            for line in gold_corpus:
                if line == "":
                    continue
                #print line
                #sys.stdout.flush()
            
                line = line.split('\t')
                #print line
                example = unicode(line[0], 'utf-8')

                label = int(line[1])
                gold_labels[example] = label
            self.gold_labels = gold_labels
            self.use_gold_extractor = False
    def delete_hits(self, task_id):
        #Delete all the fake tasks and answers we made
        print "Deleting simulated hits"
        sys.stdout.flush()


    def create_hits(self, category_id, task_id, num_hits, task_object):

        #No need to create any hits. Do the ones that were assigned to you.

        category_2_sentences = []
        category_2_worker_ids = []
        category_2_question_names = []
        
        question_names = []
        
        examples = []
        labels = []

        for hit in range(num_hits):

            worker_id = str(uuid.uuid1())
            

            if category_id == 0:
                generated_sentence = self.generate_data.pop()
                #generated_sentence = sample(self.generate_data, 1)[0]

                answer = (generated_sentence +
                          "\tTrigger\tPast\tFuture\tGeneral\tSimTaboo")

                examples.append(generated_sentence)
                labels.append(1)
                #all_answers.append([generated_sentence, 1])
            elif category_id == 1:
                

                random_negative = self.unlabeled_corpus.pop()
                answer = (random_negative + "\tNotPos\tHypOrGen\t" +
                          "no_old_sentence" + "\tSimTaboo")
                
                examples.append(random_negative)
                labels.append(0)
                #all_answers.append([random_negative, 1])
                 
        #For efficiency reasons, get the labels from NN in one batch.
        job = Job.objects.get(id=self.job_id)
        if category_id == 2:
            category_2_sentences = []
            questions = task_object['questions']
            for question in questions:
                question = question['question_data'].split('\t')
                sentence = question[-1]
                category_2_sentences.append(sentence)

            if self.use_gold_extractor:
                predicted_labels, label_probabilities = test_cnn(
                    category_2_sentences,
                    [0 for s in category_2_sentences],
                    self.model_file_name,
                    self.vocabulary)
                #predicted_labels = test(
                #                self.job_id,
                #                category_2_sentences,
                #                [0 for s in category_2_sentences])
            else:
                predicted_labels = []
                for sentence in category_2_sentences:
                    #with some probability, add some noise.
                    #print sentence
                    sentence = unicode(sentence, 'utf-8')
                    if random() < app.config['EXPERIMENT_WORKER_ACC']:
                        predicted_labels.append(self.gold_labels[sentence])
                    else:
                        predicted_labels.append(
                            1-self.gold_labels[sentence])

                print "HERE ARE THE WORKER LABELS WE ARE USING"
                print predicted_labels
                sys.stdout.flush()

            for label, sentence in zip(
                    predicted_labels,
                    category_2_sentences):
                examples.append(sentence)
                if label == 1:
                    answer = "Yes\tTrigger\tPast\tFuture\tGeneral\tHypothetical"
                    labels.append(1)
                    #all_answers.append([sentence, 1])
                else:
                    answer = "No\tFailing"
                    labels.append(0)
                    #all_answers.append([sentence, 0])
                    

        #mturk_connection_url = insert_connection_into_s3(cPickle.dumps(self))
        #job.mturk_connection = mturk_connection_url
        #job.save()
        
        return examples, labels
