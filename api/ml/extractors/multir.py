import subprocess
import os
from computeScores import computeScoresFromFile

def run_multir(training_data_file_name, testing_data_file_name,
               outputfile_name, relInd):
    FNULL = open(os.devnull, 'w')

    multi_r_command = "/homes/gws/anglil/projects/learner/code_learn/multir.sh"

    subprocess.call([multi_r_command, training_data_file_name,
                     testing_data_file_name, outputfile_name],
                    stdout=FNULL)
    
    outputfile = open(outputfile_name, 'r')
    testing_file = open(testing_data_file_name, 'r')
    
    precision, recall, f1 = computeScoresFromFile(outputfile,
                                          testing_file, relInd)

    outputfile.close()
    testing_file.close()
    FNULL.close()
    
    return precision, recall, f1
