import numpy as np
import re
import itertools
from collections import Counter
from random import shuffle
import pickle
from app import app

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

relations = ['nationality', 'born', 'lived', 'died', 'travel']

def load_test_data_and_labels(test_sentences, test_labels):

    
    test_sentences = [s.strip() for s in test_sentences]
    # Split by words
    x_text = [clean_str(sent) for sent in test_sentences]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    y = []
    for label in test_labels:
        if label == 1:
            y.append([0,1])
        else:
            y.append([1,0])
    return [x_text, y]

def load_data_and_labels(training_sentences, training_labels):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    """
    relation = relations[relInd]
    positive_training_data_file = open(training_data_file_name, 'r')
    positive_examples = []
    for line in positive_training_data_file:
        line = line.split('\t')
        sentence = line[11]
        positive_examples.append(sentence)

    positive_examples = [s.strip() for s in positive_examples]

    negative_training_data_file= open(
        '/homes/gws/chrislin/relex_data/training_data/train_CS_MJ_neg_comb_new_feature_%s' % relation, 'r')
    negative_examples = []
    for line in negative_training_data_file:
        line = line.split('\t')
        sentence = line[11]
        negative_examples.append(sentence)
    
    negative_examples = [s.strip() for s in negative_examples]

    #Set the ratio of negative to positive
    shuffle(negative_examples)
    negative_examples = negative_examples[0:len(positive_examples)]
    """
    
    # Split by words
    #x_text = positive_examples + negative_examples
    x_text = training_sentences
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels

    y = []
    for label in training_labels:
        if label == 0:
            y.append([1,0])
        elif label == 1:
            y.append([0,1])
            
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    #y = np.concatenate([positive_labels, negative_labels], 0)

    print "amount of training data:"
    print len(x_text)
    print len(y)
    
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>", sequence_length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence. If the sentence is overlength, cut it off.
    Returns padded sentences.
    """
    if not sequence_length:
        sequence_length = max(len(x) for x in sentences)
        
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence) + 1
        if num_padding < 0:
            new_sentence = sentence[0: len(sentence)+num_padding]
        else:
            new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, padding_word="<PAD/>"):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """

    x = []
    for sentence in sentences:
        sentence_embedding = []
        for word in sentence:
            if word not in vocabulary:
                sentence_embedding.append(vocabulary[padding_word])
            else:
                sentence_embedding.append(vocabulary[word])
        x.append(sentence_embedding)
        
    #x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = np.array(x)
    y = np.array(labels)

    return [x, y]


def load_data(training_sentences, training_labels):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(training_sentences,
                                             training_labels)
    sentences_padded = pad_sentences(sentences)
    sequence_length =  max(len(x) for x in sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    #remember the vocabulary that was used.

    #pickle.dump((vocabulary, vocabulary_inv, sequence_length),
    #            open('temp_data', 'wb'))

    #pickle.dump((sentences, labels),
    #            open('temp_data', 'wb'))

    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, sequence_length]

def load_test_data(test_sentences, test_labels, vocabulary):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

 
    #(train_vocabulary, train_vocabulary_inv, sequence_length) = pickle.load(
    #    open('temp_data', 'rb'))


    (train_vocabulary, train_vocabulary_inv, sequence_length) = vocabulary

    # Load and preprocess data
    test_sentences, test_labels = load_test_data_and_labels(
        test_sentences, test_labels)
    #sequence_length =  max(len(x) for x in train_sentences)
    test_sentences_padded = pad_sentences(
        test_sentences,
        sequence_length = sequence_length)
    
    #vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(test_sentences_padded,
                            test_labels, train_vocabulary)

    print "loading test data"
    print len(train_vocabulary)
    print len(train_vocabulary_inv)
    print sequence_length
    print len(x)
    print len(y)
    
    return [x, y, train_vocabulary, train_vocabulary_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
