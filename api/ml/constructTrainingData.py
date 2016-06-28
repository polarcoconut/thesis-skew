#
#
# This file takes some positive data and some negative data
# and constructs one file out of it.
#
#
#
from random import sample, shuffle
import re
from random import shuffle
import time


def constructTrainingData(positive_examples, negative_examples):

    output_training_data_file_name =  'temp_training_data'
    combined_results_clean_file = open(output_training_data_file_name, 'w')

    numPositives = 0
    numNegatives = 0
    
    examples = {}

    for example in positive_examples:
        example = example.replace("\n", "")
        example = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ",
                         example)
        combined_results_clean_file.write('%s\t1\n' % example)

        numPositives += 1
        
    for example in negative_examples:
        example = example.replace("\n", "")
        example = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ",
                         example)
        combined_results_clean_file.write('%s\t0\n' % example)

        numNegatives += 1


    return output_training_data_file_name
