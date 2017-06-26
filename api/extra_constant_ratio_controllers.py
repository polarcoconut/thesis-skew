#THIS FILE IS DEPRECATED.

def label_only_constant_ratio_controller(task_ids, task_categories,
                                        training_examples,
                                        training_labels, task_information,
                                        costSoFar, budget, job_id):

    next_category = app.config['EXAMPLE_CATEGORIES'][2]

    (selected_examples, 
     expected_labels) =  get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
        task_ids, task_categories,
        training_examples, training_labels,
        task_information, costSoFar,
        budget, job_id)


    task = make_labeling_crowdjs_task(selected_examples,
                                      expected_labels,
                                      task_information)

    return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

def label_only_US_constant_ratio_controller(
        task_ids, task_categories, training_examples,
        training_labels, task_information,
        costSoFar, budget, job_id):


    next_category = app.config['EXAMPLE_CATEGORIES'][2]


    if len(task_ids) == 0:
        
        (selected_examples, 
         expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
    else:
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
        
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']


#Alternate back and forth between precision and recall categories.
#Then, use the other half of the budget and
#select a bunch of examples from corpus to label.
#THIS CONTROLLER CAN ONLY BE USED DURING EXPERIMENTS BECAUSE IT REQUIRES
#GOLD LABELS
def seed_US_constant_ratio_controller(
        task_ids, task_categories, training_examples,
        training_labels, task_information,
        costSoFar, budget, job_id):


    print "Seed Controller activated."
    sys.stdout.flush()

    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']
        
    task_categories_per_cycle = num_negatives_wanted + 1
        
    if len(task_categories) >= (task_categories_per_cycle * 
                                num_negatives_wanted):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
    
        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    if len(task_categories) % task_categories_per_cycle == 0:

        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] *
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        
        return next_category['id'], task, num_hits, num_hits*next_category['price']


    def round_robin_constant_ratio_controller(task_ids, task_categories, 
                                          training_examples,
                                          training_labels, task_information,
                                          costSoFar, budget, job_id):


    print "RRCR Controller activated."
    sys.stdout.flush()
        
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 2

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):
        print "choosing the PRECISION category"
        sys.stdout.flush()

        
        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        

        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if (len(task_categories) % task_categories_per_cycle  == 
        num_negatives_wanted + 1):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        print "choosing the LABEL category"
        sys.stdout.flush()

        (selected_examples, 
         expected_labels) = get_unlabeled_examples_from_corpus_at_fixed_ratio(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    def round_robin_half_constant_ratio_controller(task_ids, task_categories, 
                                          training_examples,
                                          training_labels, task_information,
                                          costSoFar, budget, job_id):


    print "RR HALF CR Controller activated."
    sys.stdout.flush()
        
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 2

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):
        print "choosing the PRECISION category"
        sys.stdout.flush()

        
        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        

        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if (len(task_categories) % task_categories_per_cycle  == 
        num_negatives_wanted + 1):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        print "choosing the LABEL category"
        sys.stdout.flush()

        selected_examples = []
        expected_labels = []

        if (len(task_categories) % (task_categories_per_cycle * 2)  == 
            (task_categories_per_cycle + num_negatives_wanted + 1)):
            
            (selected_examples_2, 
             expected_labels_2) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            selected_examples += selected_examples_2
            expected_labels += expected_labels_2
            
        else:
            (selected_examples_1, 
             expected_labels_1) = get_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            selected_examples += selected_examples_1
            expected_labels += expected_labels_1
            


        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

def round_robin_US_constant_ratio_controller(task_ids, task_categories, 
                                             training_examples,
                                             training_labels, task_information,
                                             costSoFar, budget, job_id):
    

    print "RR US CR Controller activated."
    sys.stdout.flush()
        
    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

    task_categories_per_cycle = num_negatives_wanted + 2

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']


    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):
        print "choosing the PRECISION category"
        sys.stdout.flush()

        
        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
        

        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if (len(task_categories) % task_categories_per_cycle  == 
        num_negatives_wanted + 1):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        print "choosing the LABEL category"
        sys.stdout.flush()


            
        (selected_examples, 
         expected_labels) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
             task_ids, task_categories,
             training_examples, training_labels,
             task_information, costSoFar,
             budget, job_id)
        
            
    

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']

    def round_robin_constant_ratio_random_labeling_controller(
        task_ids, task_categories, 
        training_examples,
        training_labels, task_information,
        costSoFar, budget, job_id):


    print "RRCRRL Controller activated."
    sys.stdout.flush()

    if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
        num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
    else:
        num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']
    task_categories_per_cycle = num_negatives_wanted + 2
        

    if len(task_categories) % task_categories_per_cycle == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    if (len(task_categories) % task_categories_per_cycle >= 1 and 
        (len(task_categories) % task_categories_per_cycle <= 
         num_negatives_wanted)):

        last_batch = training_examples[-1]
        next_category = app.config['EXAMPLE_CATEGORIES'][1]

        task = make_precision_crowdjs_task(last_batch, task_information)

        num_hits = (
            app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
            app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])

        return next_category['id'], task, num_hits, num_hits*next_category['price']

    if (len(task_categories) % task_categories_per_cycle  == 
        num_negatives_wanted + 1):
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        (selected_examples, 
         expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
            task_ids, task_categories,
            training_examples, training_labels,
            task_information, costSoFar,
            budget, job_id)

        task = make_labeling_crowdjs_task(selected_examples,
                                          expected_labels,
                                          task_information)
 
        return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']


    def thompson_US_constant_ratio_controller(task_ids, task_categories, 
                   training_examples,
                   training_labels, task_information,
                   costSoFar,
                   extra_job_state,
                   budget, job_id):
    
    print "Thompson US Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_beta'] = {
            0 : None, 
            1 : None, 
            2 : [5,1]}
    else:
        last_action = task_categories[-1]
        
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:

            num_positives_retrieved = 0
            num_negatives_retrieved = 0

            for label in training_labels[-1]:
                if label == 1:
                    num_positives_retrieved += 1
                elif label == 0:
                    num_negatives_retrieved += 1
                else:
                    raise Exception

            extra_job_state['action_beta'][2][0] += num_positives_retrieved
            extra_job_state['action_beta'][2][1] += num_negatives_retrieved
            
            print "UPDATING THE BETA DISTRIBUTION"
            print extra_job_state['action_beta'][2]
            sys.stdout.flush()

           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute a distribution on the upper confidence bounds

    selected_action = None
    
    num_batches = len(training_examples)

    cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']

    #Draw a skew sample
    skew_sample = np.random.beta(
        extra_job_state['action_beta'][2][0],
        extra_job_state['action_beta'][2][1])
    num_of_positives_sample = skew_sample * app.config[
        'CONTROLLER_LABELING_BATCH_SIZE']

    #based on skew sample, compute cost.
    cost_of_action_2 = (
        app.config['EXAMPLE_CATEGORIES'][2]['price']  *
        app.config['CONTROLLER_LABELING_BATCH_SIZE'] /
        num_of_positives_sample)

    print "COSTS OF ACTIONS"
    print cost_of_action_0
    print skew_sample
    print num_of_positives_sample
    print cost_of_action_2
    print extra_job_state['action_beta'][2]
    sys.stdout.flush()


    if cost_of_action_0 < cost_of_action_2:
        selected_action = 0
    else:
        selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()
            
            (selected_examples, 
             expected_labels) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)

            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
        

def ucb_US_fixed_ratio_controller(task_ids, task_categories, 
                                  training_examples,
                                  training_labels, task_information,
                                  costSoFar,
                                  extra_job_state,
                                  budget, job_id):
    
    print "UCB with Active Learning Controller activated."
    sys.stdout.flush()

    if not extra_job_state:
        extra_job_state['action_counts'] = {0 : 0, 1 : 0, 2: 0}
        extra_job_state['action_mean_costs'] = {
            0 : app.config['EXAMPLE_CATEGORIES'][0]['price'], 
            1 : 0, 
            2: 0}
    else:

        last_action = task_categories[-1]
        #Update the mean costs from the last action
        extra_job_state['action_counts'][last_action] += 1
        if last_action == 2:
            
            empirical_skew = 0.0
            for label in training_labels[-1]:
                if label == 1:
                    empirical_skew += 1
            #empirical_skew /= len(training_labels[-1])

            if empirical_skew == 0:
                empirical_skew = app.config['UCB_SMOOTHING_PARAMETER']


            empirical_cost_of_positive = (
                app.config['EXAMPLE_CATEGORIES'][2]['price']  *
                app.config['CONTROLLER_LABELING_BATCH_SIZE'] / 
                empirical_skew)
                
            print "UPDATING THE EMPIRICAL SKEW"
            print empirical_skew
            print app.config['EXAMPLE_CATEGORIES'][2]['price']
            print  app.config['CONTROLLER_LABELING_BATCH_SIZE']
                                                      
            sys.stdout.flush()
            
            old_mean_cost = extra_job_state['action_mean_costs'][2]
            extra_job_state['action_mean_costs'][2] = (
                ((extra_job_state['action_counts'][2] - 1) * old_mean_cost +
                 empirical_cost_of_positive) / 
                extra_job_state['action_counts'][2])            
           
        else:
            if app.config['NUM_NEGATIVES_PER_POSITIVE'] < 0:
                num_negatives_wanted = Job.objects.get(id=job_id).dataset_skew
            else:
                num_negatives_wanted = app.config['NUM_NEGATIVES_PER_POSITIVE']

            #Count the number of times we've called the precision category
            number_of_precision_actions = 0
            i = -1
            while task_categories[i] == 1:
                number_of_precision_actions += 1
                i -= 1

            if number_of_precision_actions < num_negatives_wanted:
                print "choosing the PRECISION category"
                sys.stdout.flush()
                            
                last_batch = training_examples[-1]
                next_category = app.config['EXAMPLE_CATEGORIES'][1]
                
                task = make_precision_crowdjs_task(last_batch, 
                                                   task_information)
                
                num_hits = (
                    app.config['CONTROLLER_GENERATE_BATCH_SIZE'] * 
                    app.config['CONTROLLER_NUM_MODIFY_TASKS_PER_SENTENCE'])
                
                return next_category['id'], task, num_hits, num_hits*next_category['price'] 
            

    #Compute the upper confidence bounds

    selected_action = None
    
    if extra_job_state['action_counts'][2] == 0:
        selected_action = 2
    else:

        num_batches = len(training_examples)
        
        cost_of_action_0 = app.config['EXAMPLE_CATEGORIES'][0]['price']
        #cost_of_action_0 = (extra_job_state['action_mean_costs'][0] - 
        #                    sqrt(2.0 * log(num_batches) / 
        #                         extra_job_state['action_counts'][0]))
        c = app.config['UCB_EXPLORATION_CONSTANT']
        cost_of_action_2 = (extra_job_state['action_mean_costs'][2] - 
                            sqrt(c * log(num_batches) / 
                                 extra_job_state['action_counts'][2]))
        
        print "COSTS OF ACTIONS"
        print cost_of_action_0
        print extra_job_state['action_mean_costs'][0]
        print num_batches
        print extra_job_state['action_counts'][0]
        print "-----"
        print cost_of_action_2
        print extra_job_state['action_mean_costs'][2]
        print num_batches
        print extra_job_state['action_counts'][2]

        sys.stdout.flush()


        if cost_of_action_0 < cost_of_action_2:
            selected_action = 0
        else:
            selected_action = 2


    if selected_action == 0:
        print "choosing the RECALL category"
        sys.stdout.flush()
    
        next_category = app.config['EXAMPLE_CATEGORIES'][0]
        
        task = make_recall_crowdjs_task(task_information)
                                        
        num_hits = app.config['CONTROLLER_GENERATE_BATCH_SIZE']
        return next_category['id'], task, num_hits, num_hits * next_category['price']



    else:
        next_category = app.config['EXAMPLE_CATEGORIES'][2]
        
        if extra_job_state['action_counts'][2] == 0:
        
            (selected_examples, 
             expected_labels) = get_random_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
            
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
            
        else:            
            print "choosing the LABEL category"
            sys.stdout.flush()

            (selected_examples, 
             expected_labels) = get_US_unlabeled_examples_from_corpus_at_fixed_ratio(
                 task_ids, task_categories,
                 training_examples, training_labels,
                 task_information, costSoFar,
                 budget, job_id)
            
            task = make_labeling_crowdjs_task(selected_examples,
                                              expected_labels,
                                              task_information)
 
            return next_category['id'], task, len(selected_examples) * app.config['CONTROLLER_LABELS_PER_QUESTION'], app.config['CONTROLLER_LABELING_BATCH_SIZE'] * app.config['CONTROLLER_LABELS_PER_QUESTION'] * next_category['price']
