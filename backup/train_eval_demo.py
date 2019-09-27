
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math
import os

import numpy as np

import tensorflow as tf
import cnn_MT

BATCH_SIZE = 15 # 128  
MAX_STEPS = 10000000
TRAIN_DIR = './train'
log_device_placement = False
loss_print_interval = 1
set_save_checkpoint_secs = 600  # set to very big value(sec) to save model only when ends training
set_save_interval = 18000*5
#set_save_interval = 300000

eval_dir = './eval'
EVAL_DATA = 'test'
checkpoint_dir = './train'
eval_interval_secs = 60
num_examples = 18
run_once = True
NUM_CLASSES = 12

def eval_once(saver, summary_writer, top_k_op, top_k_predict_op, probabilities_op, summary_op, images, labels): 

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil( (num_examples * NUM_CLASSES) / BATCH_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      true_prob = 0  # Computes the probabilities of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE
      step = 0
      
      start_time = time.time()
      
      while step < num_iter and not coord.should_stop():
      
        #### original code ####        
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1
          
        ############### modified for debugging. prints predicted label ####################  
        """    
        image, label, test_labels = sess.run([images, labels, top_k_predict_op]) 
        print("\n\nNumber\t:\tTrueLabel\tPredicted Label")
        idx = 0
        while idx < BATCH_SIZE:
           print (idx, "\t:\t", int(label[idx]), "\t\t", int(test_labels[idx]))
           if  int(label[idx]) == int(test_labels[idx]):
              true_count += 1
           idx += 1
        step += 1
        """
      
      duration = time.time() - start_time
      examples_per_sec = duration/total_sample_count
      #print('examples/sec: ', examples_per_sec)
      
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      f = open('accuracy.txt', 'a')
      f.write(", precision = ")
      f.write(str(precision))
      f.write("\n")
      f.close()  

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = EVAL_DATA == 'test'
    images, labels = cnn_MT.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the inference model.
    logits = cnn_MT.inference(images, Train_Flag = False)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    ############### modified for debugging ####################
    top_k_predict_op = tf.argmax(logits,1)
    probabilities_op = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cnn_MT.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      ############### modified for debugging ####################
      eval_once(saver, summary_writer, top_k_op, top_k_predict_op, probabilities_op, summary_op, images, labels)
      if run_once:
        break
      time.sleep(eval_interval_secs)

def train_once(saver, top_k_op, top_k_predict_op,  images, labels): 

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(set_save_interval / BATCH_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      true_prob = 0  # Computes the probabilities of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE
      step = 0
      
      start_time = time.time()
      
      while step < num_iter and not coord.should_stop():
      
        #### original code ####        
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1
                
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('precision: ', precision)
      #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def train():

  with tf.Graph().as_default():

    # saved model restore
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    global_step_init = -1
    if ckpt and ckpt.model_checkpoint_path:
        global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        global_step = tf.Variable(global_step_init, name='global_step', dtype=tf.int64, trainable=False)
    else:
        global_step = tf.train.get_or_create_global_step()

    # Get images and labels
    images, labels = cnn_MT.distorted_inputs()
    print('labels:', labels)

    # Build a Graph that computes the logits predictions from the inference model.(hypothesis case)
    logits = cnn_MT.inference(images, Train_Flag = True)
    print("logits: ", logits)

    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_predict_op = tf.argmax(logits,1)
    # Calculate loss.
    loss = cnn_MT.loss(logits, labels)
    
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = cnn_MT.train(loss, global_step)
    
    class _LoggerHook(tf.train.SessionRunHook):
    
      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % loss_print_interval == 0:
          num_examples_per_step = BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          #print (format_str % (datetime.now(), self._step, loss_value,
          #                     examples_per_sec, sec_per_batch))
          fp = open('current_step.txt', 'w')
          fp.write("current_step = ")
          fp.write(str(self._step))
          fp.close()

    # saved model restore
    ckpt_tmp = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt_tmp:
      current_step = int(os.path.basename(ckpt_tmp.model_checkpoint_path).split('-')[1])
    else:
      current_step = 0
      f = open('accuracy.txt', 'w')
      
    Save_interval = int(current_step + set_save_interval/BATCH_SIZE)
    
    if current_step > 0:
      f = open('accuracy.txt', 'a')
      
    f.write("current_step = ")
    f.write(str(Save_interval))
    f.close()
    
    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=TRAIN_DIR,
        hooks=[tf.train.StopAtStepHook(last_step=Save_interval),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(log_device_placement=log_device_placement), save_checkpoint_secs = set_save_checkpoint_secs) as mon_sess:

      # saved model restore
      # ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
      # run from scratch
      if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(mon_sess, ckpt.model_checkpoint_path)
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(checkpoint_dir)
    #train_once(saver, top_k_op, top_k_predict_op, images, labels)


def main(argv=None):
  #while True:   
  for i in range(10):
    train()
    evaluate()
  


if __name__ == '__main__':
  tf.app.run()
