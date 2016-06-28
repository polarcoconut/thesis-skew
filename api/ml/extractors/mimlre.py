import subprocess
import os
from computeScores import computeScoresFromFile

def run_mimlre(training_data_file_name, testing_data_file_name,
               outputfile_name, relInd):
    FNULL = open(os.devnull, 'w')

    mimlre_command = "/homes/gws/chrislin/relex_data/extractors/mimlre.sh"

    subprocess.call([mimlre_command, training_data_file_name,
                     testing_data_file_name, outputfile_name],
                    stdout=FNULL)

    #subprocess.call([mimlre_command],
    #                stdout=FNULL)

    
    outputfile = open(outputfile_name, 'r')
    testing_file = open(testing_data_file_name, 'r')
    
    precision, recall, f1 = computeScoresFromFile(outputfile,
                                          testing_file, relInd)

    outputfile.close()
    testing_file.close()
    FNULL.close()
    
    return precision, recall, f1
