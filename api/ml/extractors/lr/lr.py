
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from random import sample, shuffle
import cPickle
import os
import uuid


import numpy as np
import re
import itertools
from collections import Counter
import pickle


def clean_str(string):
    """    
    Tokenization/string cleaning for all datasets except for SST.              
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def train_lr(training_sentences, training_labels):

    #add a dummy positive example in case there are none
    #so add a sentence that is " "
    x_text = [clean_str(sent) for sent in training_sentences + [" ", " "]]


    #x_text = [s.split(" ") for s in x_text]
 
    vectorizer = TfidfVectorizer()
    training_examples = vectorizer.fit_transform(x_text)
    
    #transformer = TfidfTransformer()
    #training_examples = transformer.fit_transform(training_examples)

    """
    #Create a bag of words
    bag_of_words = {}
    index = 0
    for sent in x_text:
        for word in sent:
            if not word in bag_of_words:
                bag_of_words[sent] = 0
                index += 1
    bag_of_words['UNK'] = index
    
    num_words = len(bag_of_words.keys())
    #Create features
    training_examples = []
    for sent in x_test:
        training_example = [0] * num_words
        for word in sent:
            training_example[bag_of_words[word]] = 1
        training_examples.append(training_example)
    """

    classifier = LogisticRegression()
    #self.classifier = LogisticRegression()                                 
    #print "TRAINING THE CLASSIFIER"
    #print training_examples
    #print training_labels

    #Add an additional label
    classifier.fit(training_examples, training_labels + [1,0])
    


    #return classifier, bag_of_words
    return classifier, vectorizer.vocabulary_

    #timestamp = str(uuid.uuid1())
    #output_file = open('runs/lr-%s' % timestamp, 'w')

    #classifier_string = cPickle.dumps(classifier)
    #output_file.write(classifier_string)
    
    #output_file.close()


def test_lr(testing_sentences, test_labels, classifier, bag_of_words):

    x_text = [clean_str(sent) for sent in testing_sentences]
    #x_text = [s.split(" ") for s in x_text]

    """
    num_words = len(bag_of_words.keys())

    testing_examples = []
    for sent in x_text:
        testing_example = [0] * num_words
        for word in sent:
            if not word in bag_of_words:
                testing_example['UNK'] = 1
            testing_example[bag_of_words[word]] = 1
        testing_examples.append(testing_example)
    """
    vectorizer = TfidfVectorizer(vocabulary = bag_of_words)
    testing_examples = vectorizer.fit_transform(x_text)

    predicted_labels = classifier.predict(testing_examples)
    predicted_probabilities = classifier.predict_proba(testing_examples)

 
    return predicted_labels, predicted_probabilities
