
from random import sample
import pickle
import sys

def greedy_controller(task_categories, training_examples,
                      training_labels, task_information,
                      config):


    sys.stdout.flush()

    #if task_categories is empty,
    #pick recall category  that doesn't require previous training data
    if len(task_categories) % 2 == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = config['EXAMPLE_CATEGORIES'][0]
        (event_name, event_definition,
         event_good_example_1,
         event_good_example_1_trigger,
         event_good_example_2,
         event_good_example_2_trigger,
         event_bad_example_1,
         event_negative_good_example_1,
         event_negative_bad_example_1) = task_information
        
        question_data = ''
        for information in task_information:
            question_data += "%s\t"
            
        question_data =  question_data % task_information
        questions = []

        #Or you can just have 1 question, and have answers_per_question be the
        #batch size
        question = {
            'question_name': 'Recall Question',
            'question_description': 'Batch %d' % (len(task_categories) + 1),
            'question_data': question_data,
            'requester_id' : config['CROWDJS_REQUESTER_ID'],
            'answers_per_question' : (config['CONTROLLER_BATCH_SIZE'] *
                                      config['CONTROLLER_APQ'])}
        questions.append(question)

        task = {'task_name': next_category['task_name'],
                'task_description': next_category['task_description'],
                'requester_id' : config['CROWDJS_REQUESTER_ID'],
                'data' : pickle.dumps({}),
                'questions' : questions,
                'unique_workers' : False}
            
        return next_category, task

    #If task_categories has one element in it, pick a category that
    #can use previous training data
    if len(task_categories) % 2 == 1:

        next_category = config['EXAMPLE_CATEGORIES'][1]

        (event_name, event_definition,
         event_good_example_1,
         event_good_example_1_trigger,
         event_good_example_2,
         event_good_example_2_trigger,
         event_bad_example_1,
         event_negative_good_example_1,
         event_negative_bad_example_1) = task_information
        
        question_data = ''
        for information in task_information:
            question_data += "%s\t"
        question_data += "%s"
        
        questions = []
        for training_example in training_examples[0]:
            new_question_information = task_information + (training_example,)
            new_question_data =  question_data % new_question_information
            question = {
                'question_name': 'Precision Question',
                'question_description': 'Batch %d' % (len(task_categories) + 1),
                'question_data': new_question_data,
                'requester_id' : config['CROWDJS_REQUESTER_ID'],
                'answers_per_question' : config['CONTROLLER_APQ']}
            questions.append(question)


        task = {'task_name': next_category['task_name'],
                'task_description': next_category['task_description'],
                'requester_id' : config['CROWDJS_REQUESTER_ID'],
                'data' : pickle.dumps({}),
                'questions' : questions}

        return next_category, task
