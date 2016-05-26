
from app import app
from random import sample

def greedy_controller(task_categories, training_examples,
                      training_labels, task_information):

    #if task_categories is empty,
    #pick recall category  that doesn't require previous training data
    if len(task_categories) % 2 == 0:

        next_category = app.EXAMPLE_CATEGORIES[0]
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

        #For the recall task, answers per question is basically the number of
        #different workers you want
        #the batch size is the number of times you want each worker to answer
        #the question
        #Alternatively, you can make answers per question = 1
        #Or you can just have 1 question, and have answers_per_question be the
        #batch size
        question = {
            'question_name': 'Recall Question %d' % i,
            'question_description': 'Batch %d' % len(task_categories) + 1,
            'question_data': question_data,
            'requester_id' : app.CROWDJS_REQUESTER_ID,
            'answers_per_question' : app.CONTROLLER_BATCH_SIZE}
        questions.append(question)

        task = {'task_name': next_category['task_name'],
                'task_description': next_category['task_description'],
                'requester_id' : app.REQUESTER_ID,
                'data' : pickle.dumps({}),
                'questions' : questions,
                'unique_workers' : False}

        return task

    #If task_categories has one element in it, pick a category that
    #can use previous training data
    if len(task_categories) % 2 == 1:

        next_category = app.EXAMPLE_CATEGORIES[1]

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
                'question_name': 'Precision Question %d' % i,
                'question_description': 'Batch %d' % len(task_categories) + 1,
                'question_data': new_question_data,
                'requester_id' : app.CROWDJS_REQUESTER_ID,
                'answers_per_question' : app.CONTROLLER_APQ}
            questions.append(question)


        task = {'task_name': next_category['task_name'],
                'task_description': next_category['task_description'],
                'requester_id' : app.REQUESTER_ID,
                'data' : pickle.dumps({}),
                'questions' : questions}

        return task
