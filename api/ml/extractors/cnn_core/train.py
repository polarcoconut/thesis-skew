#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from computeScores import computeScores
import sys
import uuid

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
#2,3,4,5,6,7
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    

def train_cnn(training_sentences, training_labels):
    # Parameters
    # ==================================================

    # Model Hyperparameters


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


    # Data Preparatopn
    # ==================================================

    # Load data
    print("Loading data...")
    sys.argv
    x, y, vocabulary, vocabulary_inv, sequence_length = data_helpers.load_data(
        training_sentences, training_labels)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    print x
    print y
    
    # Split train/dev set
    # 90/10 split

    #Assume the last tenth is the dev set.
    len_train_split = int(0.9 * len(x))
    x_train, x_dev = x[:len_train_split], x[len_train_split:]
    y_train, y_dev = y[:len_train_split], y[len_train_split:]
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Sanity Check split: {:d}/{:d}".format(len(x_train), len(x_dev)))



    # Training
    # ==================================================

    #with tf.Graph().as_default(), tf.device('/gpu:2'):
    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10,
        #                            allow_growth = True)
        gpu_options = tf.GPUOptions(allow_growth = True)
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options = gpu_options)
            #intra_op_parallelism_threads=90,
            #inter_op_parallelism_threads=90)
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = 0.01
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step,
                100, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            #timestamp = str(int(time.time()))
            timestamp = str(uuid.uuid1())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, predictions, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op,
                     cnn.predictions, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()


                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, predictions, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.predictions,
                     cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                gold_labels = []
                for gold_label in y_batch:
                    gold_labels.append(np.argmax(gold_label))
                precision, recall, fscore = computeScores(
                    predictions, gold_labels)
                print ("precision: %f, recall: %f, fscore: %f" % (
                    precision, recall, fscore))

                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                #if current_step % FLAGS.evaluate_every == 0:
                #    print("\nEvaluation:")
                #    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #    print("")
                #if current_step % FLAGS.checkpoint_every == 0:
                #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #    print("Saved model checkpoint to {}\n".format(path))
                

            while True:
                try:
                    path = saver.save(sess, checkpoint_prefix, 
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    break
                except Exception:
                    time.sleep(3600)
                    continue
                    
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            return checkpoint_file,(vocabulary, vocabulary_inv, sequence_length)
