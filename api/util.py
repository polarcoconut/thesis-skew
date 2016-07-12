import nltk
import pickle
import sys
import json
import requests
from app import app
from schema.job import Job

#old_taboo_words is a python pickle that is actually a dictionary
#mapping words to the number of times
#they have been used
@app.celery.task(name='compute_taboo_words')
def compute_taboo_words(old_taboo_words, old_sentence, new_sentence, task_id,
                        requester_id, put_task_data_url):
    nltk.download('punkt')
    nltk.download('stopwords')

    old_taboo_words = pickle.loads(old_taboo_words)
                
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

    return taboo_words
    


def write_model_to_file(job_id):
    job = Job.objects.get(id = job_id)

        
    
    
    temp_model_file_handle = open('temp_model_file', 'wb')
    temp_model_file_handle.write(job.model_file)
    temp_model_file_handle.close()
    
    
    temp_model_meta_file_handle = open('temp_model_file.meta', 'wb')
    temp_model_meta_file_handle.write(job.model_meta_file)
    temp_model_meta_file_handle.close()


    return 'temp_model_file'

