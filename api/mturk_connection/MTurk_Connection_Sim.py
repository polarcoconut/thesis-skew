from app import app
import sys
from mturk_connection.MTurk_Connection import MTurk_Connection
from api.crowdjs_util import get_next_assignment, submit_answer, get_answers
from api.util import parse_answers
from api.ml.extractors.cnn_core.train import train_cnn
from random import shuffle
from schema.experiment import Experiment
import uuid

class MTurk_Connection_Sim(MTurk_Connection):


    def __init__(experiment_id):

        experiment = Experiment.objects.get(experiment_id)
        task_ids = experiment.task_ids_for_simulation
        
        self.modify_data = {}
        self.generate_data = []
        
        generate_task_ids = task_ids[0]
        for task_id in generate_task_ids:
            examples, labels = parse_answers(task_id, 0)
            self.generate_data += examples
            for example in examples:
                self.modify_data[example] = []

        shuffle(self.generate_data)
        
        modify_task_ids = task_ids[1]
        for task_id in modify_task_ids:
            answers = get_answers(task_id)
            for answer in answers:
                value = answer['value']
                value = value.split('\t')
                sentence = value[0]
                previous_sentence_not_example_of_event = value[1]
                old_sentence = value[3]

                self.modify_data[old_sentence].append(sentence)
        
        #Train a classifier using angli's data.

        pos_training_data_file = open(
            'data/training_data/train_CS_MJ_pos_comb_new_feature_died', 'r')

        neg_training_data_file = open(
            'data/training_data/train_CS_MJ_neg_comb_new_feature_died', 'r')

        training_positive_examples = []
        training_negative_examples = []

        for line in pos_training_data_file:
            training_positive_examples = line.split('\t')[11]
        for line in neg_training_data_file:
            training_negative_examples = line.split('\t')[11]
            
        model_file_name, vocabulary = train_cnn(
            training_positive_examples + training_negative_examples,
            ([1 for e in training_positive_examples] +
             [0 for e in training_negative_examples]))

        self.model_file_name = model_file_name
        self.model_meta_file_name = "{}.meta".format(model_file_name)
        self.vocaulary = vocabulary
        
        
    def delete_hits(task_id):
        #Delete all the fake tasks and answers we made
        print "Deleting simulated hits"
        sys.stdout.flush()

        #How do we make an experimental wrapper?


    def create_hits(category_id, task_id, num_hits):

        #No need to create any hits. Do the ones that were assigned to you.

        category_2_sentences = []
        category_2_worker_ids = []
        category_2_question_names = []
        
        for hit in num_hits:

            worker_id = uuid.uuid4()
            
            next_assignment_data = get_next_assignment(task_id, worker_id)
            next_assignment_question_data = next_assignment_data[
                'question_data']
            next_assignment_question_name = next_assignment_data[
                'question_name']


            if category_id == 0:
                generated_sentence = self.generate_data.pop()
                answer = (generated_sentence +
                          "\tTrigger\tPast\tFuture\tGeneral\tSimTaboo")
                submit_answer(task_id, worker_id,
                              next_assignment_question_name,
                              answer)
            elif category_id == 1:
                old_sentence = next_assignment_data[9]
                modified_sentence = self.modify_data[old_sentence].pop()
                answer = (modified_sentence + "\tNotPos\tHypOrGen\t" +
                          old_sentence + "\tSimTaboo")
                submit_answer(task_id, worker_id,
                              next_assignment_question_name,
                              answer)
            elif category_id == 2:

                sentence = next_assignment_data[10]
                category_2_sentences.append(sentence)
                category_2_worker_ids.append(worker_id)
                category_2_question_names.append(question_name)
                
        #For efficiency reasons, get the labels from NN in one batch.
        if category_id == 2:
            predicted_labels = test_cnn([category_2_sentences],
                                        [0 for s in category_2_sentences],
                                        self.model_file_name,
                                        self.vocabulary)
            for label, question_name, worker_id in zip(
                    predicted_labels,
                    category_2_worker_ids,
                    category_2_question_names):
                if label == 1:
                    answer = "Yes\tTrigger\tPast\tFuture\tGeneral\tHypothetical"
                else:
                    answer = "No\tFailing"
                    
                submit_answer(task_id, worker_id, question_name, answer)
                              

